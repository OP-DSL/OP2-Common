#include "flang/Parser/parsing.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Parser/message.h"
#if __has_include("flang/Support/LangOptions.h")
#include "flang/Support/LangOptions.h"
#endif

#include "llvm/Support/raw_ostream.h"

// import ctype/string utilities, the small JSON buffer, the parse
// tree variant tags (variant), and the OS shims for reading stdin / picking
// a temp path from the standard library
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <process.h> // _getpid on Windows (used to name the temp file)
#else
#include <unistd.h> // ::getpid on POSIX
#endif

// namespace alias for LLVM Flang's parse-tree API
namespace fp = Fortran::parser;

/**
 * @brief Streaming JSON emitter.
 *
 * Written to avoid needing to pull a JSON library into the link.
 * The caller drives a sequence of `beginObject`/`key`/`value`/`endObject`
 * and `beginArray`/`value`/`endArray` calls and the writer is responsible
 * for correctly placing the commas.
 */
class Json {
public:
    void beginObject()
    {
        comma();
        out_ << "{";
        first_.push_back(true);
    }

    void endObject()
    {
        out_ << "}";
        first_.pop_back();
        markWrote();
    }

    void beginArray()
    {
        comma();
        out_ << "[";
        first_.push_back(true);
    }

    void endArray()
    {
        out_ << "]";
        first_.pop_back();
        markWrote();
    }

    void key(const std::string &k)
    {
        comma();
        writeString(k);
        out_ << ":";
        if (!first_.empty()) {
            first_.back() = true;
        }
    }

    void stringValue(const std::string &s)
    {
        comma();
        writeString(s);
        markWrote();
    }

    void intValue(int64_t v)
    {
        comma();
        out_ << v;
        markWrote();
    }

    void boolValue(bool b)
    {
        comma();
        out_ << (b ? "true" : "false");
        markWrote();
    }

    void nullValue()
    {
        comma();
        out_ << "null";
        markWrote();
    }

    /**
     * @brief Splice in a pre-rendered JSON fragment.
     *
     * Used to stitch together documents built with separate Json instances.
     *
     * @param jsonText Pre-rendered JSON fragment spliced in without quoting.
     */
    void rawValue(const std::string &jsonText)
    {
        comma();
        out_ << jsonText;
        markWrote();
    }

    /**
     * @brief Snapshot of the buffer.
     *
     * @return The accumulated JSON text.
     */
    std::string str() const
    {
        return out_.str();
    }

private:
    /**
     * @brief If this isn't the first value in the current container, emit a comma.
     */
    void comma()
    {
        if (!first_.empty()) {
            if (!first_.back()) {
                out_ << ",";
            }
            first_.back() = false;
        }
    }

    /**
     * @brief Mark as wrote (not first) without emitting anything.
     */
    void markWrote()
    {
        if (!first_.empty()) first_.back() = false;
    }

    /**
     * @brief String escaping, bail out to \uXXXX for any control byte there isn't a shorthand for.
     *
     * @param s String contents to escape and quote.
     * @see jsonEscape
     */
    void writeString(const std::string &s)
    {
        out_ << '"';
        for (char c : s) {
            switch (c) {
            case '"':
                out_ << "\\\"";
                break;
            case '\\':
                out_ << "\\\\";
                break;
            case '\n':
                out_ << "\\n";
                break;
            case '\r':
                out_ << "\\r";
                break;
            case '\t':
                out_ << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out_ << buf;
                } else {
                    out_ << c;
                }
            }
        }
        out_ << '"';
    }

    std::ostringstream out_;
    std::vector<bool> first_;
};

/**
 * @brief Lowercase an ASCII string.
 *
 * @param s ASCII identifier to lowercase.
 * @return String with ASCII letters converted to lowercase.
 */
static std::string toLower(std::string s)
{
    for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

/**
 * @brief Convert CharBlock to a string.
 *
 * @param src Cooked-source character range.
 * @return A copy of the cooked-source slice as a string.
 */
static std::string sourceText(fp::CharBlock src)
{
    return src.ToString();
}

static void emitExpr(Json &json, const fp::Expr &e);
static void emitActualArgs(Json &json, const std::list<fp::ActualArgSpec> &args);

/**
 * @brief Attempt to fold an integer-valued Expr to a plain int.
 *
 * @param e Fortran parse-tree expression.
 * @return The folded integer, or nullopt if the expression is not a supported integer form.
 */
static std::optional<int64_t> foldIntExpr(const fp::Expr &e);

/**
 * @brief Parse the textual representation of an integer literal as it appears in Flang's parse tree (e.g. "42", "-7", "1_8").
 *
 * @param text Integer literal spelling from the parse tree (digits, optional sign, optional kind suffix).
 * @return The parsed integer, or nullopt if `text` is not an integer literal.
 */
static std::optional<int64_t> parseIntText(const std::string &text)
{
    try {
        size_t pos = 0;
        long long v = std::stoll(text, &pos);
        // accept any trailing kind suffix like 4_ik, 1_8, 3_kind
        return static_cast<int64_t>(v);
    } catch (...) {
        return std::nullopt;
    }
}

/**
 * @brief Pull an integer out of a LiteralConstant variant.
 *
 * @param lit Parse-tree literal-constant node.
 * @return The integer value of an integer/signed-integer literal, otherwise nullopt.
 */
static std::optional<int64_t> foldLiteralConstant(const fp::LiteralConstant &lit)
{
    return std::visit([](const auto &alt) -> std::optional<int64_t> {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::IntLiteralConstant>) {
            // IntLiteralConstant is std::tuple<CharBlock, std::optional<KindParam>>
            const fp::CharBlock &cb = std::get<fp::CharBlock>(alt.t);
            return parseIntText(cb.ToString());
        } else if constexpr (std::is_same_v<T, fp::SignedIntLiteralConstant>) {
            const fp::CharBlock &cb = std::get<fp::CharBlock>(alt.t);
            return parseIntText(cb.ToString());
        } else {
            return std::nullopt;
        }
    },
        lit.u);
}

/**
 * @brief Constant folder for integer-valued Exprs.
 *
 * Only supports enough operators to recognise the kinds of expressions
 * real OP2 source code uses for things like op_decl_const sizes,
 * op_arg_dat indices and similar.
 *
 * @param e Fortran parse-tree expression.
 * @return The folded integer, or nullopt if the expression is not a supported integer form.
 */
static std::optional<int64_t> foldIntExpr(const fp::Expr &e)
{
    return std::visit([](const auto &alt) -> std::optional<int64_t> {
        using T = std::decay_t<decltype(alt)>;

        if constexpr (std::is_same_v<T, fp::LiteralConstant>) {
            return foldLiteralConstant(alt);
        } else if constexpr (std::is_same_v<T, fp::Expr::Parentheses>) {
            return foldIntExpr(alt.v.value());
        } else if constexpr (std::is_same_v<T, fp::Expr::UnaryPlus>) {
            return foldIntExpr(alt.v.value());
        } else if constexpr (std::is_same_v<T, fp::Expr::Negate>) {
            auto inner = foldIntExpr(alt.v.value());
            return inner ? std::optional<int64_t>{-*inner} : std::nullopt;
        } else if constexpr (std::is_same_v<T, fp::Expr::Add>) {
            auto l = foldIntExpr(std::get<0>(alt.t).value());
            auto r = foldIntExpr(std::get<1>(alt.t).value());
            if (l && r) return *l + *r;
            return std::nullopt;
        } else if constexpr (std::is_same_v<T, fp::Expr::Subtract>) {
            auto l = foldIntExpr(std::get<0>(alt.t).value());
            auto r = foldIntExpr(std::get<1>(alt.t).value());
            if (l && r) return *l - *r;
            return std::nullopt;
        } else if constexpr (std::is_same_v<T, fp::Expr::Multiply>) {
            auto l = foldIntExpr(std::get<0>(alt.t).value());
            auto r = foldIntExpr(std::get<1>(alt.t).value());
            if (l && r) return (*l) * (*r);
            return std::nullopt;
        } else if constexpr (std::is_same_v<T, fp::Expr::Divide>) {
            auto l = foldIntExpr(std::get<0>(alt.t).value());
            auto r = foldIntExpr(std::get<1>(alt.t).value());
            if (l && r && *r != 0) return (*l) / (*r);
            return std::nullopt;
        } else if constexpr (std::is_same_v<T, fp::Expr::Power>) {
            auto l = foldIntExpr(std::get<0>(alt.t).value());
            auto r = foldIntExpr(std::get<1>(alt.t).value());
            if (l && r && *r >= 0) {
                int64_t result = 1;
                for (int64_t i = 0; i < *r; ++i) result *= *l;
                return result;
            }
            return std::nullopt;
        } else {
            return std::nullopt;
        }
    },
        e.u);
}

/**
 * @brief Pull a bare identifier out of a Designator (such as "p_q", "OP_ID", "OP_READ").
 *
 * Only succeeds when the leaf is a single Name; anything more elaborate is reported
 * as nullopt so the caller falls through to "raw".
 *
 * @param d Parse-tree designator.
 * @return The lowercased name, or nullopt if the designator is not a single Name.
 */
static std::optional<std::string> designatorToName(const fp::Designator &d)
{
    return std::visit([](const auto &alt) -> std::optional<std::string> {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::DataRef>) {
            // DataRef can be a Name, StructureComponent, ArrayElement, etc.
            return std::visit([](const auto &inner) -> std::optional<std::string> {
                using U = std::decay_t<decltype(inner)>;
                if constexpr (std::is_same_v<U, fp::Name>) {
                    return toLower(inner.ToString());
                } else {
                    return std::nullopt;
                }
            },
                alt.u);
        } else {
            return std::nullopt;
        }
    },
        d.u);
}

/**
 * @brief View onto an expression that looks like a call, with the callee identifier and a
 * borrowed pointer into the parse tree's argument list.
 *
 * @see exprAsCall
 */
struct CallView {
    std::string name;
    const std::list<fp::ActualArgSpec> *args = nullptr;
};

/**
 * @brief Try to extract (callee-name, args) out of an Expr that looks like a call.
 *
 * @param e Fortran parse-tree expression.
 * @return Callee name and argument list, or nullopt if `e` is not a call-shaped expression.
 */
static std::optional<CallView> exprAsCall(const fp::Expr &e)
{
    return std::visit([](const auto &alt) -> std::optional<CallView> {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::FunctionReference>>) {
            const fp::FunctionReference &fr = alt.value();
            const fp::Call &call = fr.v;
            const fp::ProcedureDesignator &pd = std::get<fp::ProcedureDesignator>(call.t);
            const auto &args = std::get<std::list<fp::ActualArgSpec>>(call.t);
            return std::visit([&](const auto &p) -> std::optional<CallView> {
                using P = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<P, fp::Name>) {
                    return CallView{toLower(p.ToString()), &args};
                } else {
                    return std::nullopt;
                }
            },
                pd.u);
        } else if constexpr (std::is_same_v<T, fp::StructureConstructor>) {
            return std::nullopt;
        } else {
            return std::nullopt;
        }
    },
        e.u);
}

/**
 * @brief Emit one expression as a JSON object.
 *
 * @param json JSON writer to append into.
 * @param e Fortran parse-tree expression.
 */
static void emitExpr(Json &json, const fp::Expr &e)
{
    // 1. Integer literal
    if (auto v = foldIntExpr(e)) {
        json.beginObject();
        json.key("kind");
        json.stringValue("int");
        json.key("value");
        json.intValue(*v);
        json.endObject();
        return;
    }

    // 2. Character literal
    auto extractCharLiteral = [](const fp::Expr &expr) -> std::optional<std::string> {
        return std::visit([](const auto &alt) -> std::optional<std::string> {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::CharLiteralConstant>>) {
                return std::get<std::string>(alt.value().t);
            } else if constexpr (std::is_same_v<T, fp::LiteralConstant>) {
                return std::visit([](const auto &inner) -> std::optional<std::string> {
                    using U = std::decay_t<decltype(inner)>;
                    if constexpr (std::is_same_v<U, fp::CharLiteralConstant>) {
                        return std::get<std::string>(inner.t);
                    } else {
                        return std::nullopt;
                    }
                },
                    alt.u);
            } else {
                return std::nullopt;
            }
        },
            expr.u);
    };
    if (auto s = extractCharLiteral(e)) {
        json.beginObject();
        json.key("kind");
        json.stringValue("string");
        json.key("value");
        json.stringValue(*s);
        json.endObject();
        return;
    }

    // 3. Bare identifier (such as OP_READ, OP_ID, p_q)
    bool emitted = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Designator>>) {
            if (auto name = designatorToName(alt.value())) {
                json.beginObject();
                json.key("kind");
                json.stringValue("name");
                json.key("value");
                json.stringValue(*name);
                json.endObject();
                return true;
            }
        }
        return false;
    },
        e.u);
    if (emitted) return;

    // 4. Nested call (such as op_arg_dat(...), op_arg_gbl(...), op_arg_idx(...))
    if (auto call = exprAsCall(e)) {
        json.beginObject();
        json.key("kind");
        json.stringValue("call");
        json.key("name");
        json.stringValue(call->name);
        json.key("args");
        emitActualArgs(json, *call->args);
        json.endObject();
        return;
    }

    // 5. Fallback: emit the raw source text so the Python side can either
    //              parse it or flag it as unsupported
    json.beginObject();
    json.key("kind");
    json.stringValue("raw");
    json.key("source");
    json.stringValue(sourceText(e.source));
    json.endObject();
}

/**
 * @brief Emit a parenthesised argument list as a JSON array of expression objects.
 *
 * @param json JSON writer to append into.
 * @param args Actual-argument list from the call.
 */
static void emitActualArgs(Json &json, const std::list<fp::ActualArgSpec> &args)
{
    json.beginArray();
    for (const fp::ActualArgSpec &spec : args) {
        // ActualArgSpec = std::tuple<std::optional<Keyword>, ActualArg>
        const fp::ActualArg &aa = std::get<fp::ActualArg>(spec.t);
        // ActualArg variant: Indirection<Expr>, AltReturnSpec, ActualArgProcedureComponentRef, ProcedureName
        bool handled = std::visit([&](const auto &alt) -> bool {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Expr>>) {
                emitExpr(json, alt.value());
                return true;
            }
            return false;
        },
            aa.u);
        if (!handled) {
            json.beginObject();
            json.key("kind");
            json.stringValue("raw");
            json.key("source");
            json.stringValue("<unsupported-actual-arg>");
            json.endObject();
        }
    }
    json.endArray();
}

/**
 * @brief Per-subprogram dependency walker.
 *
 * Walks one subprogram subtree and gathers the lowercased names of
 * everything that looks like a call to another subroutine or function.
 * A superset is collected, so Python can filter the results.
 */
struct DependsCollector {
    std::set<std::string> &out;

    /**
     * @brief Direct subroutine call.
     *
     * @param cs Parse-tree CALL statement.
     * @return Always `true`, so Flang continues walking the subtree.
     */
    bool Pre(const fp::CallStmt &cs)
    {
        const fp::Call &c = std::get<fp::Call>(cs.t);
        const fp::ProcedureDesignator &pd = std::get<fp::ProcedureDesignator>(c.t);
        std::visit([&](const auto &p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, fp::Name>) {
                out.insert(toLower(p.ToString()));
            }
        },
            pd.u);
        return true;
    }

    /**
     * @brief Function-style reference inside an expression.
     *
     * May be a real function call or array indexing; Python disambiguates.
     *
     * @param fr Parse-tree function reference (or could be array indexing).
     * @return Always `true`, so Flang continues walking the subtree.
     */
    bool Pre(const fp::FunctionReference &fr)
    {
        const fp::Call &c = fr.v;
        const fp::ProcedureDesignator &pd = std::get<fp::ProcedureDesignator>(c.t);
        std::visit([&](const auto &p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<T, fp::Name>) {
                out.insert(toLower(p.ToString()));
            }
        },
            pd.u);
        return true;
    }

    // no-op fallbacks for every other parse-tree node type
    template <typename T>
    bool Pre(const T &)
    {
        return true;
    }
    template <typename T>
    void Post(const T &)
    {}
};

static void emitBodyExpr(Json &json, const fp::Expr &e);

/**
 * @brief Render a KindParam as its raw source text.
 *
 * @param kp Kind parameter (`_kind` on a literal or type spec).
 * @return Raw source spelling of the kind selector.
 */
static std::string kindParamToString(const fp::KindParam &kp)
{
    return std::visit([](const auto &alt) -> std::string {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, std::uint64_t>) {
            return std::to_string(alt);
        } else {
            // Scalar<Integer<Constant<Name>>>
            return toLower(alt.thing.thing.thing.ToString());
        }
    },
        kp.u);
}

/**
 * @brief Emit one of the four literal-constant leaf shapes (int/real/logical/char), or "unsupported".
 *
 * @param json JSON writer to append into.
 * @param lit Parse-tree literal-constant node.
 */
static void emitLiteralConstant(Json &json, const fp::LiteralConstant &lit)
{
    bool emitted = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::IntLiteralConstant>) {
            const auto &cb = std::get<fp::CharBlock>(alt.t);
            const auto &kindOpt = std::get<std::optional<fp::KindParam>>(alt.t);
            json.beginObject();
            json.key("kind");
            json.stringValue("int_lit");
            json.key("text");
            json.stringValue(sourceText(cb));
            json.key("kind_text");
            if (kindOpt) json.stringValue(kindParamToString(*kindOpt));
            else json.nullValue();
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::RealLiteralConstant>) {
            const auto &real = std::get<fp::RealLiteralConstant::Real>(alt.t);
            const auto &kindOpt = std::get<std::optional<fp::KindParam>>(alt.t);
            json.beginObject();
            json.key("kind");
            json.stringValue("real_lit");
            json.key("text");
            json.stringValue(sourceText(real.source));
            json.key("kind_text");
            if (kindOpt) json.stringValue(kindParamToString(*kindOpt));
            else json.nullValue();
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::LogicalLiteralConstant>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("logical_lit");
            json.key("value");
            json.boolValue(std::get<bool>(alt.t));
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::CharLiteralConstant>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("char_lit");
            json.key("value");
            json.stringValue(std::get<std::string>(alt.t));
            json.endObject();
            return true;
        }
        return false;
    },
        lit.u);

    if (!emitted) {
        json.beginObject();
        json.key("kind");
        json.stringValue("unsupported");
        json.key("tag");
        json.stringValue("literal_constant");
        json.endObject();
    }
}

/**
 * @brief Emit one array subscript as JSON, either an IntExpr, or a SubscriptTriplet.
 *
 * @param json JSON writer to append into.
 * @param sub Parse-tree section-subscript (scalar index or triplet).
 */
static void emitBodySubscript(Json &json, const fp::SectionSubscript &sub)
{
    std::visit([&](const auto &alt) {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::SubscriptTriplet>) {
            const auto &t = alt.t;
            auto emitBound = [&](const char *key, const std::optional<fp::Subscript> &bound) {
                json.key(key);
                if (bound) {
                    emitBodyExpr(json, bound->thing.thing.value());
                } else {
                    json.nullValue();
                }
            };
            json.beginObject();
            json.key("kind");
            json.stringValue("triplet");
            emitBound("lower", std::get<0>(t));
            emitBound("upper", std::get<1>(t));
            emitBound("stride", std::get<2>(t));
            json.endObject();
        } else {
            // Subscript = ScalarIntExpr = Scalar<Integer<Indirection<Expr>>>.
            emitBodyExpr(json, alt.thing.value());
        }
    },
        sub.u);
}

/**
 * @brief Emit a Designator.
 *
 * Only structurally decomposes the two shapes the validator cares about:
 * (plain Name, and array-element via a plain-Name base).
 * Everything else becomes "raw".
 *
 * @param json JSON writer to append into.
 * @param d Parse-tree designator.
 */
static void emitDesignator(Json &json, const fp::Designator &d)
{
    bool emitted = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::DataRef>) {
            return std::visit([&](const auto &inner) -> bool {
                using U = std::decay_t<decltype(inner)>;
                if constexpr (std::is_same_v<U, fp::Name>) {
                    json.beginObject();
                    json.key("kind");
                    json.stringValue("name");
                    json.key("value");
                    json.stringValue(toLower(inner.ToString()));
                    json.endObject();
                    return true;
                } else if constexpr (std::is_same_v<U, Fortran::common::Indirection<fp::ArrayElement>>) {
                    const fp::ArrayElement &ae = inner.value();
                    return std::visit([&](const auto &baseAlt) -> bool {
                        using V = std::decay_t<decltype(baseAlt)>;
                        if constexpr (std::is_same_v<V, fp::Name>) {
                            json.beginObject();
                            json.key("kind");
                            json.stringValue("part_ref");
                            json.key("name");
                            json.stringValue(toLower(baseAlt.ToString()));
                            json.key("subscripts");
                            json.beginArray();
                            for (const auto &s : ae.Subscripts()) emitBodySubscript(json, s);
                            json.endArray();
                            json.endObject();
                            return true;
                        }
                        return false;
                    },
                        ae.Base().u);
                } else {
                    return false;
                }
            },
                alt.u);
        }
        return false;
    },
        d.u);

    if (!emitted) {
        json.beginObject();
        json.key("kind");
        json.stringValue("raw");
        json.key("source");
        json.stringValue(sourceText(d.source));
        json.endObject();
    }
}

/**
 * @brief Emit a FunctionReference.
 *
 * @param json JSON writer to append into.
 * @param fr Parse-tree function reference (may also be array indexing).
 */
static void emitFuncRef(Json &json, const fp::FunctionReference &fr)
{
    const fp::Call &call = fr.v;
    const fp::ProcedureDesignator &pd = std::get<fp::ProcedureDesignator>(call.t);
    const auto &args = std::get<std::list<fp::ActualArgSpec>>(call.t);

    std::optional<std::string> name = std::visit([](const auto &p) -> std::optional<std::string> {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, fp::Name>) {
            return toLower(p.ToString());
        } else {
            return std::nullopt;
        }
    },
        pd.u);

    if (!name) {
        json.beginObject();
        json.key("kind");
        json.stringValue("raw");
        json.key("source");
        json.stringValue("<complex-procedure-designator>");
        json.endObject();
        return;
    }

    json.beginObject();
    json.key("kind");
    json.stringValue("funcref");
    json.key("name");
    json.stringValue(*name);
    json.key("args");
    json.beginArray();
    for (const fp::ActualArgSpec &spec : args) {
        const fp::ActualArg &aa = std::get<fp::ActualArg>(spec.t);
        bool handled = std::visit([&](const auto &alt) -> bool {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Expr>>) {
                emitBodyExpr(json, alt.value());
                return true;
            }
            return false;
        },
            aa.u);
        if (!handled) {
            json.beginObject();
            json.key("kind");
            json.stringValue("raw");
            json.key("source");
            json.stringValue("<unsupported-actual-arg>");
            json.endObject();
        }
    }
    json.endArray();
    json.endObject();
}

/**
 * @brief Emit a Variable.
 *
 * Used for the LHS of an AssignmentStmt.
 *
 * @param json JSON writer to append into.
 * @param v Parse-tree variable (designator or function-reference).
 */
static void emitVariable(Json &json, const fp::Variable &v)
{
    std::visit([&](const auto &alt) {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Designator>>) {
            emitDesignator(json, alt.value());
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::FunctionReference>>) {
            emitFuncRef(json, alt.value());
        }
    },
        v.u);
}

/**
 * @brief Source range of an assignment LHS, used for diagnostic line numbers.
 *
 * Prefer the designator (or function-reference name) over the RHS, so a
 * continued assignment reports the line of `lhs =` rather than the last
 * continued RHS fragment.
 */
static fp::CharBlock variableSource(const fp::Variable &v)
{
    return std::visit([](const auto &alt) -> fp::CharBlock {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Designator>>) {
            return alt.value().source;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::FunctionReference>>) {
            const fp::ProcedureDesignator &pd = std::get<fp::ProcedureDesignator>(alt.value().v.t);
            fp::CharBlock src;
            std::visit([&](const auto &p) {
                using U = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<U, fp::Name>) {
                    src = p.source;
                }
            },
                pd.u);
            return src;
        }
        return fp::CharBlock{};
    },
        v.u);
}

static fp::CharBlock assignmentLineSource(const fp::AssignmentStmt &assign)
{
    const auto &lhs = std::get<fp::Variable>(assign.t);
    const auto &rhs = std::get<fp::Expr>(assign.t);
    fp::CharBlock src = variableSource(lhs);
    return src.empty() ? rhs.source : src;
}

static void emitBodyExpr(Json &json, const fp::Expr &e)
{
    bool emitted = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;

        if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Designator>>) {
            emitDesignator(json, alt.value());
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::FunctionReference>>) {
            emitFuncRef(json, alt.value());
            return true;
        } else if constexpr (std::is_same_v<T, fp::Expr::Parentheses>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("paren");
            json.key("expr");
            emitBodyExpr(json, alt.v.value());
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::Expr::UnaryPlus>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("unary");
            json.key("op");
            json.stringValue("+");
            json.key("expr");
            emitBodyExpr(json, alt.v.value());
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::Expr::Negate>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("unary");
            json.key("op");
            json.stringValue("-");
            json.key("expr");
            emitBodyExpr(json, alt.v.value());
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::Expr::Add> || std::is_same_v<T, fp::Expr::Subtract> ||
                             std::is_same_v<T, fp::Expr::Multiply> || std::is_same_v<T, fp::Expr::Divide> ||
                             std::is_same_v<T, fp::Expr::Power> || std::is_same_v<T, fp::Expr::Concat> ||
                             std::is_same_v<T, fp::Expr::LT> || std::is_same_v<T, fp::Expr::LE> ||
                             std::is_same_v<T, fp::Expr::EQ> || std::is_same_v<T, fp::Expr::NE> ||
                             std::is_same_v<T, fp::Expr::GE> || std::is_same_v<T, fp::Expr::GT> ||
                             std::is_same_v<T, fp::Expr::AND> || std::is_same_v<T, fp::Expr::OR> ||
                             std::is_same_v<T, fp::Expr::EQV> || std::is_same_v<T, fp::Expr::NEQV>) {
            const char *op = "+";
            if constexpr (std::is_same_v<T, fp::Expr::Subtract>) op = "-";
            else if constexpr (std::is_same_v<T, fp::Expr::Multiply>) op = "*";
            else if constexpr (std::is_same_v<T, fp::Expr::Divide>) op = "/";
            else if constexpr (std::is_same_v<T, fp::Expr::Power>) op = "**";
            else if constexpr (std::is_same_v<T, fp::Expr::Concat>) op = "//";
            else if constexpr (std::is_same_v<T, fp::Expr::LT>) op = "<";
            else if constexpr (std::is_same_v<T, fp::Expr::LE>) op = "<=";
            else if constexpr (std::is_same_v<T, fp::Expr::EQ>) op = "==";
            else if constexpr (std::is_same_v<T, fp::Expr::NE>) op = "!=";
            else if constexpr (std::is_same_v<T, fp::Expr::GE>) op = ">=";
            else if constexpr (std::is_same_v<T, fp::Expr::GT>) op = ">";
            else if constexpr (std::is_same_v<T, fp::Expr::AND>) op = "&&";
            else if constexpr (std::is_same_v<T, fp::Expr::OR>) op = "||";
            else if constexpr (std::is_same_v<T, fp::Expr::EQV>) op = "==";
            else if constexpr (std::is_same_v<T, fp::Expr::NEQV>) op = "!=";

            json.beginObject();
            json.key("kind");
            json.stringValue("binary");
            json.key("op");
            json.stringValue(op);
            json.key("left");
            emitBodyExpr(json, std::get<0>(alt.t).value());
            json.key("right");
            emitBodyExpr(json, std::get<1>(alt.t).value());
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::Expr::NOT>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("unary");
            json.key("op");
            json.stringValue("!");
            json.key("expr");
            emitBodyExpr(json, alt.v.value());
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::LiteralConstant>) {
            emitLiteralConstant(json, alt);
            return true;
        }

        return false;
    },
        e.u);

    if (emitted) return;

    // anything else that isn't decomposed becomes an opaque "unsupported"
    // leaf carrying the source text, so the Python side can raise a clear
    // error rather than silently misreading it as a value
    json.beginObject();
    json.key("kind");
    json.stringValue("unsupported");
    json.key("tag");
    json.stringValue("expr");
    json.key("source");
    json.stringValue(sourceText(e.source));
    json.endObject();
}

/**
 * @brief Parse-tree walker that appends every Name it visits, lowercased.
 */
struct NameCollector {
    std::vector<std::string> &out; // lowercased names, in visit order
    bool Pre(const fp::Name &n)
    {
        out.push_back(toLower(n.ToString()));
        return true;
    }
    template <typename T>
    bool Pre(const T &)
    {
        return true;
    }
    template <typename T>
    void Post(const T &)
    {}
};

/**
 * @brief Parse-tree walker that adds every local array declaration into a JSON as {"name", "dims"}. 
 */
struct LocalsCollector {
    Json &json; // emits directly into an open array of {"name", "dims"} objects

    static const fp::ArraySpec *findArraySpecAttr(const std::list<fp::AttrSpec> &attrs)
    {
        for (const auto &attr : attrs) {
            if (const auto *spec = std::get_if<fp::ArraySpec>(&attr.u)) return spec;
        }
        return nullptr;
    }

    static std::vector<std::string> collectShapeDimNames(const fp::ArraySpec *spec)
    {
        std::vector<std::string> names;
        if (!spec) return names;
        if (const auto *shapes = std::get_if<std::list<fp::ExplicitShapeSpec>>(&spec->u)) {
            NameCollector nc{names};
            for (const auto &shape : *shapes) fp::Walk(shape, nc);
        }
        return names;
    }

    bool Pre(const fp::TypeDeclarationStmt &decl)
    {
        const auto &attrs = std::get<std::list<fp::AttrSpec>>(decl.t);
        const fp::ArraySpec *attrArraySpec = findArraySpecAttr(attrs);

        const auto &entityDecls = std::get<std::list<fp::EntityDecl>>(decl.t);
        for (const auto &entityDecl : entityDecls) {
            const fp::Name &nameNode = std::get<fp::ObjectName>(entityDecl.t);
            const auto &ownSpec = std::get<std::optional<fp::ArraySpec>>(entityDecl.t);

            const fp::ArraySpec *spec = ownSpec ? &*ownSpec : attrArraySpec;
            if (!spec) continue;

            std::vector<std::string> dims = collectShapeDimNames(spec);

            json.beginObject();
            json.key("name");
            json.stringValue(toLower(nameNode.ToString()));
            json.key("dims");
            json.beginArray();
            for (const auto &d : dims) json.stringValue(d);
            json.endArray();
            json.endObject();
        }
        return true;
    }

    template <typename T>
    bool Pre(const T &)
    {
        return true;
    }
    template <typename T>
    void Post(const T &)
    {}
};

/**
 * @brief Per-subprogram assignment/call walker.
 *
 * Walks one subprogram's Execution_Part and records every assignment
 * statement (lhs/rhs expr trees) and every direct subroutine call.
 */
struct BodyCollector {
    Json &jsonAssignments; // open array of {"line", "lhs", "rhs"}
    Json &jsonCalls;       // open array of {"line", "name", "args"}
    const fp::AllCookedSources &cooked;

    std::pair<int, int> resolveLineCol(fp::CharBlock src) const
    {
        if (src.empty()) return {0, 0};
        auto prov = cooked.GetProvenanceRange(src);
        if (!prov) return {0, 0};
        auto pos = cooked.allSources().GetSourcePosition(prov->start());
        if (pos) return {static_cast<int>(pos->line), static_cast<int>(pos->column)};
        return {0, 0};
    }

    bool Pre(const fp::AssignmentStmt &assign)
    {
        const auto &lhs = std::get<fp::Variable>(assign.t);
        const auto &rhs = std::get<fp::Expr>(assign.t);
        auto [line, col] = resolveLineCol(assignmentLineSource(assign));

        jsonAssignments.beginObject();
        jsonAssignments.key("line");
        jsonAssignments.intValue(line);
        jsonAssignments.key("lhs");
        emitVariable(jsonAssignments, lhs);
        jsonAssignments.key("rhs");
        emitBodyExpr(jsonAssignments, rhs);
        jsonAssignments.endObject();
        return true;
    }

    bool Pre(const fp::CallStmt &call)
    {
        const fp::Call &c = std::get<fp::Call>(call.t);
        const fp::ProcedureDesignator &pd = std::get<fp::ProcedureDesignator>(c.t);
        const auto &args = std::get<std::list<fp::ActualArgSpec>>(c.t);

        std::string name;
        fp::CharBlock nameSrc;
        bool gotName = std::visit([&](const auto &alt) -> bool {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, fp::Name>) {
                name = toLower(alt.ToString());
                nameSrc = alt.source;
                return true;
            }
            return false;
        },
            pd.u);

        if (!gotName) return true;

        auto [line, col] = resolveLineCol(nameSrc);

        jsonCalls.beginObject();
        jsonCalls.key("line");
        jsonCalls.intValue(line);
        jsonCalls.key("name");
        jsonCalls.stringValue(name);
        jsonCalls.key("args");
        jsonCalls.beginArray();
        for (const fp::ActualArgSpec &spec : args) {
            const fp::ActualArg &aa = std::get<fp::ActualArg>(spec.t);
            bool handled = std::visit([&](const auto &alt) -> bool {
                using T = std::decay_t<decltype(alt)>;
                if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Expr>>) {
                    emitBodyExpr(jsonCalls, alt.value());
                    return true;
                }
                return false;
            },
                aa.u);
            if (!handled) {
                jsonCalls.beginObject();
                jsonCalls.key("kind");
                jsonCalls.stringValue("raw");
                jsonCalls.key("source");
                jsonCalls.stringValue("<unsupported-actual-arg>");
                jsonCalls.endObject();
            }
        }
        jsonCalls.endArray();
        jsonCalls.endObject();
        return true;
    }

    template <typename T>
    bool Pre(const T &)
    {
        return true;
    }
    template <typename T>
    void Post(const T &)
    {}
};

// Parse-tree unwrap helpers
static const fp::Expr &unwrapScalarIntExpr(const fp::ScalarIntExpr &e)
{
    return e.thing.thing.value();
}
static const fp::Expr &unwrapScalarLogicalExpr(const fp::ScalarLogicalExpr &e)
{
    return e.thing.thing.value();
}
static const fp::Expr &unwrapScalarExpr(const fp::ScalarExpr &e)
{
    return e.thing.value();
}
static const fp::Expr &unwrapConstantExpr(const fp::ConstantExpr &e)
{
    return e.thing.value();
}
static const fp::Expr &unwrapScalarIntConstantExpr(const fp::ScalarIntConstantExpr &e)
{
    return e.thing.thing.thing.value();
}
static const fp::Expr &unwrapSpecificationExpr(const fp::SpecificationExpr &e)
{
    return unwrapScalarIntExpr(e.v);
}

/**
 * @brief Render a KindSelector as its raw source text, or nullopt if absent or not a ScalarIntConstantExpr.
 *
 * @param ks Optional kind selector on an intrinsic type spec.
 * @return Kind-selector source text, or nullopt if absent or of an unsupported form.
 */
static std::optional<std::string> kindSelectorText(const std::optional<fp::KindSelector> &ks)
{
    if (!ks) return std::nullopt;
    if (const auto *sice = std::get_if<fp::ScalarIntConstantExpr>(&ks->u)) {
        return sourceText(unwrapScalarIntConstantExpr(*sice).source);
    }
    return std::nullopt;
}

/**
 * @brief Emit a TypeParamValue as a scalar int expression, or "unsupported" for Star/Deferred.
 *
 * @param json JSON writer to append into.
 * @param tpv Character length type-param-value.
 */
static void emitTypeParamValue(Json &json, const fp::TypeParamValue &tpv)
{
    if (const auto *sie = std::get_if<fp::ScalarIntExpr>(&tpv.u)) {
        emitBodyExpr(json, unwrapScalarIntExpr(*sie));
        return;
    }
    json.beginObject();
    json.key("kind");
    json.stringValue("unsupported");
    json.key("tag");
    json.stringValue("char_length");
    json.endObject();
}

/**
 * @brief Emit a CharLength; a TypeParamValue or integer literal.
 *
 * @param json JSON writer to append into.
 * @param cl Parse-tree character length (`*n` or a type-param-value).
 */
static void emitCharLength(Json &json, const fp::CharLength &cl)
{
    if (const auto *tpv = std::get_if<fp::TypeParamValue>(&cl.u)) {
        emitTypeParamValue(json, *tpv);
        return;
    }
    json.beginObject();
    json.key("kind");
    json.stringValue("int_lit");
    json.key("text");
    json.stringValue(std::to_string(std::get<std::uint64_t>(cl.u)));
    json.key("kind_text");
    json.nullValue();
    json.endObject();
}

/**
 * @brief Emit the length part of an optional CharSelector.
 *
 * @param json JSON writer to append into.
 * @param cs Optional character selector (`*n` or `(LEN=..., KIND=...)`).
 */
static void emitCharLen(Json &json, const std::optional<fp::CharSelector> &cs)
{
    if (!cs) {
        json.nullValue();
        return;
    }

    std::visit([&](const auto &alt) {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::LengthSelector>) {
            std::visit([&](const auto &inner) {
                using U = std::decay_t<decltype(inner)>;
                if constexpr (std::is_same_v<U, fp::TypeParamValue>) {
                    emitTypeParamValue(json, inner);
                } else {
                    emitCharLength(json, inner);
                }
            },
                alt.u);
        } else {
            const auto &lengthOpt = std::get<0>(alt.t);
            if (lengthOpt) emitTypeParamValue(json, *lengthOpt);
            else json.nullValue();
        }
    },
        cs->u);
}

/**
 * @brief Emit an intrinsic type spec as {"kind": "intrinsic", "base", "kind_text", "charlen"}, or "unsupported".
 *
 * @param json JSON writer to append into.
 * @param its Intrinsic type spec.
 */
static void emitIntrinsicType(Json &json, const fp::IntrinsicTypeSpec &its)
{
    std::visit([&](const auto &alt) {
        using T = std::decay_t<decltype(alt)>;
        json.beginObject();
        if constexpr (std::is_same_v<T, fp::IntegerTypeSpec> ||
                      std::is_same_v<T, fp::IntrinsicTypeSpec::Real> ||
                      std::is_same_v<T, fp::IntrinsicTypeSpec::Logical>) {
            json.key("kind");
            json.stringValue("intrinsic");
            json.key("base");
            json.stringValue(
                std::is_same_v<T, fp::IntegerTypeSpec> ? "integer" : std::is_same_v<T, fp::IntrinsicTypeSpec::Real> ? "real"
                                                                                                                    : "logical");
            auto kt = kindSelectorText(alt.v);
            json.key("kind_text");
            if (kt) json.stringValue(*kt);
            else json.nullValue();
            json.key("charlen");
            json.nullValue();
        } else if constexpr (std::is_same_v<T, fp::IntrinsicTypeSpec::Character>) {
            json.key("kind");
            json.stringValue("intrinsic");
            json.key("base");
            json.stringValue("character");
            json.key("kind_text");
            json.nullValue();
            json.key("charlen");
            emitCharLen(json, alt.v);
        } else {
            json.key("kind");
            json.stringValue("unsupported");
        }
        json.endObject();
    },
        its.u);
}

/**
 * @brief Emit an DeclarationTypeSpec.
 *
 * @param json JSON writer to append into.
 * @param dts Declaration type spec.
 */
static void emitDeclType(Json &json, const fp::DeclarationTypeSpec &dts)
{
    if (const auto *its = std::get_if<fp::IntrinsicTypeSpec>(&dts.u)) {
        emitIntrinsicType(json, *its);
        return;
    }
    json.beginObject();
    json.key("kind");
    json.stringValue("unsupported");
    json.endObject();
}

/**
 * @brief Emit an explicit-shape ArraySpec as {"kind": "explicit", "shape": [{lb, ub}, ...]}, or "unsupported".
 *
 * @param json JSON writer to append into.
 * @param spec Array spec from a type-decl attribute or entity suffix.
 */
static void emitArraySpec(Json &json, const fp::ArraySpec &spec)
{
    const auto *shapes = std::get_if<std::list<fp::ExplicitShapeSpec>>(&spec.u);
    if (!shapes) {
        json.beginObject();
        json.key("kind");
        json.stringValue("unsupported");
        json.endObject();
        return;
    }

    json.beginObject();
    json.key("kind");
    json.stringValue("explicit");
    json.key("shape");
    json.beginArray();
    for (const fp::ExplicitShapeSpec &dim : *shapes) {
        const auto &lbOpt = std::get<0>(dim.t);
        const auto &ub = std::get<1>(dim.t);

        json.beginObject();
        json.key("lb");
        if (lbOpt) emitBodyExpr(json, unwrapSpecificationExpr(*lbOpt));
        else json.nullValue();
        json.key("ub");
        emitBodyExpr(json, unwrapSpecificationExpr(ub));
        json.endObject();
    }
    json.endArray();
    json.endObject();
}

/**
 * @brief Emit an entity initializer as a constant expression.
 *
 * @param json JSON writer to append into.
 * @param init Optional entity initializer.
 */
static void emitInitialization(Json &json, const std::optional<fp::Initialization> &init)
{
    if (!init) {
        json.nullValue();
        return;
    }
    if (const auto *ce = std::get_if<fp::ConstantExpr>(&init->u)) {
        emitBodyExpr(json, unwrapConstantExpr(*ce));
        return;
    }

    json.beginObject();
    json.key("kind");
    json.stringValue("unsupported");
    json.key("tag");
    json.stringValue("initialization");
    json.endObject();
}

static bool hasParameterAttr(const std::list<fp::AttrSpec> &attrs)
{
    for (const fp::AttrSpec &attr : attrs) {
        if (std::holds_alternative<fp::Parameter>(attr.u)) return true;
    }
    return false;
}

/**
 * @brief Emit a DATA-statement constant.
 *
 * @param json JSON writer to append into.
 * @param dc DATA-statement constant.
 */
static void emitDataStmtConstant(Json &json, const fp::DataStmtConstant &dc)
{
    bool emitted = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::LiteralConstant>) {
            emitLiteralConstant(json, alt);
            return true;
        } else if constexpr (std::is_same_v<T, fp::SignedIntLiteralConstant>) {
            const auto &cb = std::get<fp::CharBlock>(alt.t);
            const auto &kindOpt = std::get<std::optional<fp::KindParam>>(alt.t);
            json.beginObject();
            json.key("kind");
            json.stringValue("int_lit");
            json.key("text");
            json.stringValue(sourceText(cb));
            json.key("kind_text");
            if (kindOpt) json.stringValue(kindParamToString(*kindOpt));
            else json.nullValue();
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, fp::SignedRealLiteralConstant>) {
            const auto &signOpt = std::get<0>(alt.t);
            const auto &real = std::get<fp::RealLiteralConstant>(alt.t);
            const auto &realCore = std::get<fp::RealLiteralConstant::Real>(real.t);
            const auto &kindOpt = std::get<std::optional<fp::KindParam>>(real.t);
            std::string text = sourceText(realCore.source);
            if (signOpt && *signOpt == fp::Sign::Negative) text = "-" + text;
            json.beginObject();
            json.key("kind");
            json.stringValue("real_lit");
            json.key("text");
            json.stringValue(text);
            json.key("kind_text");
            if (kindOpt) json.stringValue(kindParamToString(*kindOpt));
            else json.nullValue();
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Designator>>) {
            emitDesignator(json, alt.value());
            return true;
        }
        return false;
    },
        dc.u);

    if (!emitted) {
        json.beginObject();
        json.key("kind");
        json.stringValue("unsupported");
        json.key("tag");
        json.stringValue("data_stmt_constant");
        json.endObject();
    }
}

/**
 * @brief Emit a DATA statement as {"kind": "data_stmt", "sets": [{objects, values}, ...]}.
 *
 * @param json JSON writer to append into.
 * @param dstmt DATA statement.
 */
static void emitDataStmtNode(Json &json, const fp::DataStmt &dstmt)
{
    json.beginObject();
    json.key("kind");
    json.stringValue("data_stmt");
    json.key("sets");
    json.beginArray();
    for (const fp::DataStmtSet &set : dstmt.v) {
        const auto &objects = std::get<std::list<fp::DataStmtObject>>(set.t);
        const auto &values = std::get<std::list<fp::DataStmtValue>>(set.t);

        json.beginObject();

        json.key("objects");
        json.beginArray();
        for (const fp::DataStmtObject &obj : objects) {
            std::visit([&](const auto &alt) {
                using T = std::decay_t<decltype(alt)>;
                if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Variable>>) {
                    emitVariable(json, alt.value());
                } else {
                    // DataImpliedDo - not used by OP2 kernels.
                    json.beginObject();
                    json.key("kind");
                    json.stringValue("unsupported");
                    json.key("tag");
                    json.stringValue("data_implied_do");
                    json.endObject();
                }
            },
                obj.u);
        }
        json.endArray();

        json.key("values");
        json.beginArray();
        for (const fp::DataStmtValue &val : values) {
            const auto &repeatOpt = std::get<std::optional<fp::DataStmtRepeat>>(val.t);
            const auto &constant = std::get<fp::DataStmtConstant>(val.t);

            json.beginObject();
            json.key("repeated");
            json.boolValue(repeatOpt.has_value());
            json.key("value");
            emitDataStmtConstant(json, constant);
            json.endObject();
        }
        json.endArray();

        json.endObject();
    }
    json.endArray();
    json.endObject();
}

/**
 * @brief Per-subprogram specification-part walker that emits type_decl, parameter_stmt, and data_stmt objects.
 */
struct DeclCollector {
    Json &json;

    static const fp::ArraySpec *findArraySpecAttr(const std::list<fp::AttrSpec> &attrs)
    {
        for (const fp::AttrSpec &attr : attrs) {
            if (const auto *spec = std::get_if<fp::ArraySpec>(&attr.u)) return spec;
        }
        return nullptr;
    }

    bool Pre(const fp::TypeDeclarationStmt &decl)
    {
        const auto &declTypeSpec = std::get<fp::DeclarationTypeSpec>(decl.t);
        const auto &attrs = std::get<std::list<fp::AttrSpec>>(decl.t);
        const auto &entityDecls = std::get<std::list<fp::EntityDecl>>(decl.t);

        const fp::ArraySpec *attrArraySpec = findArraySpecAttr(attrs);

        json.beginObject();
        json.key("kind");
        json.stringValue("type_decl");
        json.key("type");
        emitDeclType(json, declTypeSpec);
        json.key("is_parameter");
        json.boolValue(hasParameterAttr(attrs));
        json.key("dim");
        if (attrArraySpec) emitArraySpec(json, *attrArraySpec);
        else json.nullValue();

        json.key("entities");
        json.beginArray();
        for (const fp::EntityDecl &ed : entityDecls) {
            const fp::Name &nameNode = std::get<fp::ObjectName>(ed.t);
            const auto &ownSpec = std::get<std::optional<fp::ArraySpec>>(ed.t);
            const auto &init = std::get<std::optional<fp::Initialization>>(ed.t);

            json.beginObject();
            json.key("name");
            json.stringValue(toLower(nameNode.ToString()));
            json.key("dim");
            if (ownSpec) emitArraySpec(json, *ownSpec);
            else json.nullValue();
            json.key("init");
            emitInitialization(json, init);
            json.endObject();
        }
        json.endArray();

        json.endObject();
        return true;
    }

    bool Pre(const fp::ParameterStmt &pstmt)
    {
        json.beginObject();
        json.key("kind");
        json.stringValue("parameter_stmt");
        json.key("defs");
        json.beginArray();
        for (const fp::NamedConstantDef &def : pstmt.v) {
            const fp::NamedConstant &nc = std::get<fp::NamedConstant>(def.t);
            const fp::ConstantExpr &ce = std::get<fp::ConstantExpr>(def.t);

            json.beginObject();
            json.key("name");
            json.stringValue(toLower(nc.v.ToString()));
            json.key("value");
            emitBodyExpr(json, unwrapConstantExpr(ce));
            json.endObject();
        }
        json.endArray();
        json.endObject();
        return true;
    }

    bool Pre(const fp::DataStmt &dstmt)
    {
        emitDataStmtNode(json, dstmt);
        return true;
    }

    template <typename T>
    bool Pre(const T &)
    {
        return true;
    }
    template <typename T>
    void Post(const T &)
    {}
};

static void emitBlock(Json &json, const fp::Block &block, const fp::AllCookedSources &cooked);

static std::pair<int, int> resolveLineColStmt(const fp::AllCookedSources &cooked, fp::CharBlock src)
{
    if (src.empty()) return {0, 0};
    auto prov = cooked.GetProvenanceRange(src);
    if (!prov) return {0, 0};
    auto pos = cooked.allSources().GetSourcePosition(prov->start());
    if (pos) return {static_cast<int>(pos->line), static_cast<int>(pos->column)};
    return {0, 0};
}

/**
 * @brief Emit a CALL statement as {"kind": "call", "line", "name", "args"}, or "unsupported" if the callee isn't a plain Name.
 *
 * @param json JSON writer to append into.
 * @param call Parse-tree CALL statement.
 * @param cooked Flang cooked-source map used to recover original line/column.
 */
static void emitCallStmtNode(Json &json, const fp::CallStmt &call, const fp::AllCookedSources &cooked)
{
    const fp::Call &c = std::get<fp::Call>(call.t);
    const fp::ProcedureDesignator &pd = std::get<fp::ProcedureDesignator>(c.t);
    const auto &args = std::get<std::list<fp::ActualArgSpec>>(c.t);

    std::string name;
    fp::CharBlock nameSrc;
    bool gotName = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::Name>) {
            name = toLower(alt.ToString());
            nameSrc = alt.source;
            return true;
        }
        return false;
    },
        pd.u);

    if (!gotName) {
        json.beginObject();
        json.key("kind");
        json.stringValue("unsupported");
        json.key("tag");
        json.stringValue("call_stmt");
        json.endObject();
        return;
    }

    auto [line, col] = resolveLineColStmt(cooked, nameSrc);

    json.beginObject();
    json.key("kind");
    json.stringValue("call");
    json.key("line");
    json.intValue(line);
    json.key("name");
    json.stringValue(name);
    json.key("args");
    json.beginArray();
    for (const fp::ActualArgSpec &spec : args) {
        const fp::ActualArg &aa = std::get<fp::ActualArg>(spec.t);
        bool handled = std::visit([&](const auto &alt) -> bool {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::Expr>>) {
                emitBodyExpr(json, alt.value());
                return true;
            }
            return false;
        },
            aa.u);
        if (!handled) {
            json.beginObject();
            json.key("kind");
            json.stringValue("unsupported");
            json.key("tag");
            json.stringValue("actual_arg");
            json.endObject();
        }
    }
    json.endArray();
    json.endObject();
}

/**
 * @brief Emit an action statement (assign/call/continue/if_stmt/return/stop/write), or "unsupported".
 *
 * @param json JSON writer to append into.
 * @param a Action statement.
 * @param cooked Flang cooked-source map used to recover original line/column.
 */
static void emitActionStmt(Json &json, const fp::ActionStmt &a, const fp::AllCookedSources &cooked)
{
    bool emitted = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;

        if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::AssignmentStmt>>) {
            const fp::AssignmentStmt &as = alt.value();
            const auto &lhs = std::get<fp::Variable>(as.t);
            const auto &rhs = std::get<fp::Expr>(as.t);
            auto [line, col] = resolveLineColStmt(cooked, assignmentLineSource(as));

            json.beginObject();
            json.key("kind");
            json.stringValue("assign");
            json.key("line");
            json.intValue(line);
            json.key("lhs");
            emitVariable(json, lhs);
            json.key("rhs");
            emitBodyExpr(json, rhs);
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::CallStmt>>) {
            emitCallStmtNode(json, alt.value(), cooked);
            return true;
        } else if constexpr (std::is_same_v<T, fp::ContinueStmt>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("continue");
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::IfStmt>>) {
            const fp::IfStmt &ifs = alt.value();
            const auto &cond = std::get<fp::ScalarLogicalExpr>(ifs.t);
            const auto &inner = std::get<fp::UnlabeledStatement<fp::ActionStmt>>(ifs.t);

            json.beginObject();
            json.key("kind");
            json.stringValue("if_stmt");
            json.key("cond");
            emitBodyExpr(json, unwrapScalarLogicalExpr(cond));
            json.key("stmt");
            emitActionStmt(json, inner.statement, cooked);
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::ReturnStmt>>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("return");
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::StopStmt>>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("stop");
            json.endObject();
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::WriteStmt>>) {
            json.beginObject();
            json.key("kind");
            json.stringValue("write");
            json.endObject();
            return true;
        }

        return false;
    },
        a.u);

    if (!emitted) {
        json.beginObject();
        json.key("kind");
        json.stringValue("unsupported");
        json.key("tag");
        json.stringValue("action_stmt");
        json.endObject();
    }
}

/**
 * @brief Emit an IF construct as {"kind": "if_construct", "branches": [{cond, body}, ...]}.
 * 
 * `else` has cond=null.
 *
 * @param json JSON writer to append into.
 * @param ifc IF construct.
 * @param cooked Flang cooked-source map used to recover original line/column.
 */
static void emitIfConstruct(Json &json, const fp::IfConstruct &ifc, const fp::AllCookedSources &cooked)
{
    const auto &ifThen = std::get<fp::Statement<fp::IfThenStmt>>(ifc.t);
    const auto &thenBlock = std::get<fp::Block>(ifc.t);
    const auto &elseIfBlocks = std::get<std::list<fp::IfConstruct::ElseIfBlock>>(ifc.t);
    const auto &elseBlockOpt = std::get<std::optional<fp::IfConstruct::ElseBlock>>(ifc.t);

    json.beginObject();
    json.key("kind");
    json.stringValue("if_construct");
    json.key("branches");
    json.beginArray();

    json.beginObject();
    json.key("cond");
    emitBodyExpr(json, unwrapScalarLogicalExpr(std::get<fp::ScalarLogicalExpr>(ifThen.statement.t)));
    json.key("body");
    emitBlock(json, thenBlock, cooked);
    json.endObject();

    for (const fp::IfConstruct::ElseIfBlock &eib : elseIfBlocks) {
        const auto &stmt = std::get<fp::Statement<fp::ElseIfStmt>>(eib.t);
        const auto &blk = std::get<fp::Block>(eib.t);

        json.beginObject();
        json.key("cond");
        emitBodyExpr(json, unwrapScalarLogicalExpr(std::get<fp::ScalarLogicalExpr>(stmt.statement.t)));
        json.key("body");
        emitBlock(json, blk, cooked);
        json.endObject();
    }

    if (elseBlockOpt) {
        const auto &blk = std::get<fp::Block>(elseBlockOpt->t);

        json.beginObject();
        json.key("cond");
        json.nullValue();
        json.key("body");
        emitBlock(json, blk, cooked);
        json.endObject();
    }

    json.endArray();
    json.endObject();
}

/**
 * @brief Emit a DO construct as counted (`do i = lb, ub[, step]`) or while.
 * 
 * `DO CONCURRENT` becomes "unsupported".
 *
 * @param json JSON writer to append into.
 * @param dc DO construct.
 * @param cooked Flang cooked-source map used to recover original line/column.
 */
static void emitDoConstruct(Json &json, const fp::DoConstruct &dc, const fp::AllCookedSources &cooked)
{
    const auto &doStmt = std::get<fp::Statement<fp::NonLabelDoStmt>>(dc.t);
    const auto &block = std::get<fp::Block>(dc.t);
    const auto &loopControlOpt = std::get<std::optional<fp::LoopControl>>(doStmt.statement.t);

    json.beginObject();
    json.key("kind");
    json.stringValue("do");

    bool handled = false;
    if (loopControlOpt) {
        handled = std::visit([&](const auto &alt) -> bool {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, fp::LoopControl::Bounds>) {
                json.key("mode");
                json.stringValue("counted");
                json.key("var");
                json.stringValue(toLower(alt.Name().thing.ToString()));
                json.key("lb");
                emitBodyExpr(json, unwrapScalarExpr(alt.Lower()));
                json.key("ub");
                emitBodyExpr(json, unwrapScalarExpr(alt.Upper()));
                json.key("step");
                if (alt.Step()) emitBodyExpr(json, unwrapScalarExpr(*alt.Step()));
                else json.nullValue();
                return true;
            } else if constexpr (std::is_same_v<T, fp::ScalarLogicalExpr>) {
                json.key("mode");
                json.stringValue("while");
                json.key("cond");
                emitBodyExpr(json, unwrapScalarLogicalExpr(alt));
                return true;
            }
            return false;
        },
            loopControlOpt->u);
    }

    if (!handled) {
        json.key("mode");
        json.stringValue("unsupported");
    }

    json.key("body");
    emitBlock(json, block, cooked);
    json.endObject();
}

/**
 * @brief Emit one execution-part construct; an executable construct, DATA statement, or "unsupported".
 *
 * @param json JSON writer to append into.
 * @param epc Execution-part construct.
 * @param cooked Flang cooked-source map used to recover original line/column.
 */
static void emitExecutionPartConstruct(Json &json, const fp::ExecutionPartConstruct &epc, const fp::AllCookedSources &cooked);

/**
 * @brief Emit an executable construct (action stmt, IF construct, or DO construct), or "unsupported".
 *
 * @param json JSON writer to append into.
 * @param ec Executable construct.
 * @param cooked Flang cooked-source map used to recover original line/column.
 */
static void emitExecutableConstruct(Json &json, const fp::ExecutableConstruct &ec, const fp::AllCookedSources &cooked)
{
    bool emitted = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::Statement<fp::ActionStmt>>) {
            emitActionStmt(json, alt.statement, cooked);
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::IfConstruct>>) {
            emitIfConstruct(json, alt.value(), cooked);
            return true;
        } else if constexpr (std::is_same_v<T, Fortran::common::Indirection<fp::DoConstruct>>) {
            emitDoConstruct(json, alt.value(), cooked);
            return true;
        }
        return false;
    },
        ec.u);

    if (!emitted) {
        json.beginObject();
        json.key("kind");
        json.stringValue("unsupported");
        json.key("tag");
        json.stringValue("executable_construct");
        json.endObject();
    }
}

static void emitExecutionPartConstruct(Json &json, const fp::ExecutionPartConstruct &epc, const fp::AllCookedSources &cooked)
{
    bool emitted = std::visit([&](const auto &alt) -> bool {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, fp::ExecutableConstruct>) {
            emitExecutableConstruct(json, alt, cooked);
            return true;
        } else if constexpr (std::is_same_v<T, fp::Statement<Fortran::common::Indirection<fp::DataStmt>>>) {
            emitDataStmtNode(json, alt.statement.value());
            return true;
        }
        return false;
    },
        epc.u);

    if (!emitted) {
        json.beginObject();
        json.key("kind");
        json.stringValue("unsupported");
        json.key("tag");
        json.stringValue("execution_part_construct");
        json.endObject();
    }
}

/**
 * @brief Emit a Block as a JSON array of execution-part constructs.
 *
 * @param json JSON writer to append into.
 * @param block Execution-part block (list of constructs).
 * @param cooked Flang cooked-source map used to recover original line/column.
 */
static void emitBlock(Json &json, const fp::Block &block, const fp::AllCookedSources &cooked)
{
    json.beginArray();
    for (const fp::ExecutionPartConstruct &epc : block) emitExecutionPartConstruct(json, epc, cooked);
    json.endArray();
}

/**
 * @brief Top-level parse-tree visitor.
 *
 * One Scanner instance is created per file and handed to Flang's Walk(),
 * which invokes the appropriate Pre()/Post() overload for every parse-tree
 * node it visits; the templated fallbacks at the bottom of the struct make
 * sure unrecognised node types are silently traversed.
 *
 * Each successful Pre() emits zero or one JSON event into the open `events`
 * array. The walk continues into the subtree (returning true) in every case
 * so that, for example, `op_par_loop` calls inside a subroutine body are
 * still discovered.
 *
 * The events emitted are:
 *
 *   * Whenever a CallStmt callee matches `op_par_loop_<N>` -> "op_par_loop_N"
 *     event with its full argument tree.
 *   * Whenever a CallStmt callee matches `op_decl_const`   -> "op_decl_const"
 *     event with its full argument tree.
 *   * For every SubroutineSubprogram                       -> "subroutine_subprogram"
 *     event with name, parameters, depends, and source body text.
 *   * For every FunctionSubprogram                         -> "function_subprogram"
 *     event (same shape as subroutine).
 *
 * All of these go into a single ordered events array; the Python side
 * dispatches on the "kind" field.
 *
 * @see DependsCollector
 * @see LocalsCollector
 * @see BodyCollector
 * @see DeclCollector
 */
struct Scanner {
    Json &json;                         // open events array we append into
    const fp::AllCookedSources &cooked; // for mapping CharBlocks -> line/col

    /**
     * @brief Triggered for every `call ...(...)` statement.
     *
     * @param call Parse-tree CALL statement.
     * @return Always `true`, so Flang continues walking the subtree.
     */
    bool Pre(const fp::CallStmt &call)
    {
        const fp::Call &c = std::get<fp::Call>(call.t);
        const fp::ProcedureDesignator &pd = std::get<fp::ProcedureDesignator>(c.t);
        const auto &actualArgs = std::get<std::list<fp::ActualArgSpec>>(c.t);

        // events only emitted for plain-Name callees
        std::string name;
        fp::CharBlock nameSrc;
        bool gotName = std::visit([&](const auto &alt) -> bool {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, fp::Name>) {
                name = toLower(alt.ToString());
                nameSrc = alt.source;
                return true;
            } else {
                return false;
            }
        },
            pd.u);

        if (!gotName) return true;

        static const std::regex parLoopRe{"^op_par_loop_[0-9]+$"};
        const bool isParLoop = std::regex_match(name, parLoopRe);
        const bool isDeclConst = (name == "op_decl_const");
        if (!isParLoop && !isDeclConst) return true;

        auto [line, col] = resolveLineCol(nameSrc);

        if (isParLoop) {
            emitLoop(name, line, col, actualArgs);
        } else {
            emitConst(line, col, actualArgs);
        }
        return true;
    }

    /**
     * @brief Map a CharBlock from the cooked source stream back to a (line, column) in the original source file.
     *
     * @param src Cooked-source character range.
     * @return `(line, column)` in the original source; `(0, 0)` on failure.
     */
    std::pair<int, int> resolveLineCol(fp::CharBlock src)
    {
        if (src.empty()) return {0, 0};
        auto prov = cooked.GetProvenanceRange(src);
        if (!prov) return {0, 0};
        auto pos = cooked.allSources().GetSourcePosition(prov->start());
        if (pos) {
            return {static_cast<int>(pos->line), static_cast<int>(pos->column)};
        }
        return {0, 0};
    }

    // Per-event emitters
    void emitLoop(const std::string &name, int line, int col,
        const std::list<fp::ActualArgSpec> &args)
    {
        json.beginObject();
        json.key("kind");
        json.stringValue(name);
        json.key("location");
        {
            json.beginObject();
            json.key("line");
            json.intValue(line);
            json.key("column");
            json.intValue(col);
            json.endObject();
        }
        json.key("args");
        emitActualArgs(json, args);
        json.endObject();
    }

    void emitConst(int line, int col, const std::list<fp::ActualArgSpec> &args)
    {
        json.beginObject();
        json.key("kind");
        json.stringValue("op_decl_const");
        json.key("location");
        {
            json.beginObject();
            json.key("line");
            json.intValue(line);
            json.key("column");
            json.intValue(col);
            json.endObject();
        }
        json.key("args");
        emitActualArgs(json, args);
        json.endObject();
    }

    // Subprogram events

    /**
     * @brief Build a CharBlock spanning two ranges that belong to the same parse-tree subprogram.
     *
     * @param a Start of the cooked-source span (typically the opening statement).
     * @param b End of the cooked-source span (typically the END statement).
     * @return A CharBlock covering `[a.begin(), b.end())`.
     */
    static fp::CharBlock spanningRange(fp::CharBlock a, fp::CharBlock b)
    {
        if (a.empty()) return b;
        if (b.empty()) return a;
        const char *start = a.begin();
        const char *end = b.begin() + b.size();
        if (end <= start) return a;
        return fp::CharBlock{start, static_cast<std::size_t>(end - start)};
    }

    bool Pre(const fp::SubroutineSubprogram &sub)
    {
        // SubroutineSubprogram::t =
        //   tuple< Statement<SubroutineStmt>, SpecificationPart,
        //          ExecutionPart, optional<InternalSubprogramPart>,
        //          Statement<EndSubroutineStmt> >
        const auto &startStmt = std::get<fp::Statement<fp::SubroutineStmt>>(sub.t);
        const auto &endStmt = std::get<fp::Statement<fp::EndSubroutineStmt>>(sub.t);
        const fp::SubroutineStmt &subStmt = startStmt.statement;

        // SubroutineStmt::t =
        //   tuple< list<PrefixSpec>, Name, list<DummyArg>,
        //          optional<LanguageBindingSpec>, ... >
        const fp::Name &nameNode = std::get<fp::Name>(subStmt.t);
        std::string name = toLower(nameNode.ToString());
        auto [line, col] = resolveLineCol(nameNode.source);

        std::vector<std::string> parameters;
        const auto &dummyArgs = std::get<std::list<fp::DummyArg>>(subStmt.t);
        for (const auto &arg : dummyArgs) {
            std::visit([&](const auto &inner) {
                using T = std::decay_t<decltype(inner)>;
                if constexpr (std::is_same_v<T, fp::Name>) {
                    parameters.push_back(toLower(inner.ToString()));
                }
            },
                arg.u);
        }

        // walk the subprogram's subtree to collect candidate dependency names
        std::set<std::string> depends;
        DependsCollector dc{depends};
        fp::Walk(sub, dc);
        depends.erase(name);

        emitSubprogram("subroutine_subprogram", name, line, col,
            parameters, depends,
            spanningRange(startStmt.source, endStmt.source),
            std::get<fp::SpecificationPart>(sub.t),
            std::get<fp::ExecutionPart>(sub.t),
            /*fnStmt=*/nullptr);
        return true;
    }

    bool Pre(const fp::FunctionSubprogram &fn)
    {
        // FunctionSubprogram::t =
        //   tuple< Statement<FunctionStmt>, SpecificationPart,
        //          ExecutionPart, optional<InternalSubprogramPart>,
        //          Statement<EndFunctionStmt> >
        const auto &startStmt = std::get<fp::Statement<fp::FunctionStmt>>(fn.t);
        const auto &endStmt = std::get<fp::Statement<fp::EndFunctionStmt>>(fn.t);
        const fp::FunctionStmt &fnStmt = startStmt.statement;

        // FunctionStmt::t =
        //   tuple< list<PrefixSpec>, Name, list<Name>, optional<Suffix> >
        const fp::Name &nameNode = std::get<fp::Name>(fnStmt.t);
        std::string name = toLower(nameNode.ToString());
        auto [line, col] = resolveLineCol(nameNode.source);

        std::vector<std::string> parameters;
        const auto &paramList = std::get<std::list<fp::Name>>(fnStmt.t);
        for (const auto &n : paramList) {
            parameters.push_back(toLower(n.ToString()));
        }

        std::set<std::string> depends;
        DependsCollector dc{depends};
        fp::Walk(fn, dc);
        depends.erase(name);

        emitSubprogram("function_subprogram", name, line, col,
            parameters, depends,
            spanningRange(startStmt.source, endStmt.source),
            std::get<fp::SpecificationPart>(fn.t),
            std::get<fp::ExecutionPart>(fn.t),
            &fnStmt);
        return true;
    }

    /**
     * @brief Get the result name from a FunctionStmt.
     *
     * @param fnStmt Function statement whose RESULT clause is read.
     * @return lowercased RESULT name, or nullopt if no RESULT clause was written.
     */
    static std::optional<std::string> resultName(const fp::FunctionStmt &fnStmt)
    {
        const auto &suffixOpt = std::get<std::optional<fp::Suffix>>(fnStmt.t);
        if (!suffixOpt) return std::nullopt;
        const auto &nameOpt = std::get<std::optional<fp::Name>>(suffixOpt->t);
        if (!nameOpt) return std::nullopt;
        return toLower(nameOpt->ToString());
    }

    /**
     * @brief Emit the Function_Stmt's prefix return type.
     *
     * @param json JSON writer to append the type node (or JSON null) into.
     * @param fnStmt Function statement whose prefix type, if any, is emitted.
     */
    static void resultType(Json &json, const fp::FunctionStmt &fnStmt)
    {
        const auto &prefixes = std::get<std::list<fp::PrefixSpec>>(fnStmt.t);
        for (const fp::PrefixSpec &spec : prefixes) {
            if (const auto *dts = std::get_if<fp::DeclarationTypeSpec>(&spec.u)) {
                emitDeclType(json, *dts);
                return;
            }
        }
        json.nullValue();
    }

    /**
     * @brief Emit a subprogram event.
     *
     * @param kind JSON event kind string.
     * @param name lowercased identifier.
     * @param line Source line (1-based), or 0 if unknown.
     * @param col Source column (1-based), or 0 if unknown.
     * @param parameters Dummy argument names.
     * @param depends Callee names collected from the subprogram body.
     * @param bodyRange Cooked-source span covering the whole subprogram.
     * @param spec Specification part (declarations).
     * @param exec Execution part of the subprogram.
     * @param fnStmt Function statement; non-null only for function_subprogram events.
     * @see LocalsCollector
     * @see BodyCollector
     * @see DeclCollector
     */
    void emitSubprogram(const std::string &kind,
        const std::string &name,
        int line, int col,
        const std::vector<std::string> &parameters,
        const std::set<std::string> &depends,
        fp::CharBlock bodyRange,
        const fp::SpecificationPart &spec,
        const fp::ExecutionPart &exec,
        const fp::FunctionStmt *fnStmt)
    {
        json.beginObject();
        json.key("kind");
        json.stringValue(kind);
        json.key("name");
        json.stringValue(name);
        json.key("location");
        {
            json.beginObject();
            json.key("line");
            json.intValue(line);
            json.key("column");
            json.intValue(col);
            json.endObject();
        }
        json.key("parameters");
        {
            json.beginArray();
            for (const auto &p : parameters) json.stringValue(p);
            json.endArray();
        }
        json.key("depends");
        {
            json.beginArray();
            for (const auto &d : depends) json.stringValue(d);
            json.endArray();
        }
        json.key("source");
        if (bodyRange.size() > 0) {
            json.stringValue(std::string(bodyRange.begin(), bodyRange.size()));
        } else {
            json.stringValue("");
        }

        json.key("locals");
        {
            json.beginArray();
            LocalsCollector lc{json};
            fp::Walk(spec, lc);
            json.endArray();
        }

        Json assignmentsJson, callsJson;
        assignmentsJson.beginArray();
        callsJson.beginArray();
        BodyCollector bc{assignmentsJson, callsJson, cooked};
        fp::Walk(exec, bc);
        assignmentsJson.endArray();
        callsJson.endArray();

        json.key("assignments");
        json.rawValue(assignmentsJson.str());
        json.key("calls");
        json.rawValue(callsJson.str());

        // full typed declarations and a nested statement tree
        json.key("decls");
        {
            json.beginArray();
            DeclCollector dc{json};
            fp::Walk(spec, dc);
            json.endArray();
        }

        json.key("stmts");
        emitBlock(json, exec.v, cooked);

        if (fnStmt != nullptr) {
            json.key("result_name");
            auto rn = resultName(*fnStmt);
            if (rn) json.stringValue(*rn);
            else json.nullValue();

            json.key("result_type");
            resultType(json, *fnStmt);
        }

        json.endObject();
    }

    // no-op fallbacks
    template <typename T>
    bool Pre(const T &)
    {
        return true;
    }
    template <typename T>
    void Post(const T &)
    {}
};

/**
 * @brief Read all of stdin into a string.
 *
 * @return The entire stdin stream as a string.
 */
static std::string slurpStdin()
{
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

/**
 * @brief Write `contents` to a uniquely-named temp file and return its path.
 *
 * @param contents Source text to write to the temp file.
 * @param preferredDir Directory to try first (usually the original source directory).
 * @param uniqueSuffix Disambiguator so concurrent `--batch` units do not collide.
 * @return Filesystem path of the written temp file. Calls `std::exit(1)` if no temp file can be created.
 */
static std::string writeTempFile(const std::string &contents,
    const std::string &preferredDir = {},
    int uniqueSuffix = 0)
{
    namespace fs = std::filesystem;
    const std::string name = "op2-flang-scan-" + std::to_string(::getpid()) +
                             "-" + std::to_string(uniqueSuffix) + ".F90";

    auto tryWrite = [&](const fs::path &dir) -> std::optional<std::string> {
        std::error_code ec;
        if (!dir.empty() && !fs::is_directory(dir, ec)) {
            return std::nullopt;
        }
        auto path = (dir.empty() ? fs::temp_directory_path() : dir) / name;
        std::ofstream ofs(path);
        if (!ofs) {
            return std::nullopt;
        }
        ofs << contents;
        ofs.close();
        if (!ofs) {
            std::error_code removeEc;
            fs::remove(path, removeEc);
            return std::nullopt;
        }
        return path.string();
    };

    if (!preferredDir.empty()) {
        if (auto p = tryWrite(fs::path(preferredDir))) {
            return *p;
        }
    }
    if (auto p = tryWrite(fs::temp_directory_path())) {
        return *p;
    }
    std::cerr << "op2-flang-scan: failed to create temporary source file\n";
    std::exit(1);
}

using Clock = std::chrono::steady_clock;

static double msSince(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/**
 * @brief Get the parent directory of `path`.
 *
 * @param path Path whose parent directory is returned.
 * @return Parent directory of `path`, or empty if there is none.
 */
static std::string parentDirOf(const std::string &path)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path p = fs::absolute(fs::path(path), ec);
    if (ec) {
        p = fs::path(path);
    }
    auto parent = p.parent_path();
    if (parent.empty()) {
        return {};
    }
    return parent.string();
}

/**
 * @brief Escape a string for embedding in a JSON error object.
 *
 * @param s String to escape for a error-object payload.
 * @return String with JSON structural characters escaped.
 * @see Json::writeString
 */
static std::string jsonEscape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

/**
 * @brief Parse one translation unit and write one JSON object (plus newline) to stdout.
 *
 * On failure still emits a JSON object with an "error" field so --batch callers can
 * fall back per file.
 *
 * @param reportedPath Path string to put in the JSON `path` field.
 * @param onDiskPath Real file to parse when `sourceBytes` is empty.
 * @param sourceBytes In-memory source; if non-empty, materialised next to `reportedPath`.
 * @param includeDirs Extra directories for Fortran INCLUDE resolution.
 * @param emitTiming If true, print OP2_FLANG_SCAN_TIMING lines to stderr.
 * @param tempSuffix Suffix for the stdin temp-file name.
 * @param tSession0 Session start time for cumulative timing.
 * @return 0 on success, 1 on parse failure.
 * @see Scanner
 * @see writeTempFile
 */
static int scanOneUnit(const std::string &reportedPath,
    const std::string &onDiskPath,
    const std::string &sourceBytes,
    const std::vector<std::string> &includeDirs,
    bool emitTiming,
    int tempSuffix,
    Clock::time_point tSession0)
{
    const auto tUnit = Clock::now();
    std::string path = onDiskPath;
    std::string originalPath = reportedPath.empty() ? onDiskPath : reportedPath;
    std::string sourceDir = parentDirOf(originalPath);
    std::string tempFile;
    double materializeMs = 0.0;

    if (!sourceBytes.empty()) {
        const auto t0 = Clock::now();
        tempFile = writeTempFile(sourceBytes, sourceDir, tempSuffix);
        path = tempFile;
        materializeMs = msSince(t0);
    }
    if (originalPath.empty()) {
        originalPath = path;
    }
    if (sourceDir.empty()) {
        sourceDir = parentDirOf(originalPath);
    }

    fp::Options options;
    options.isFixedForm = false;
    if (!sourceDir.empty()) {
        options.searchDirectories.push_back(sourceDir);
    }
    for (const auto &dir : includeDirs) {
        options.searchDirectories.push_back(dir);
    }

    fp::AllSources allSources;
    fp::AllCookedSources cooked{allSources};
    fp::Parsing parsing{cooked};

    const auto tParse0 = Clock::now();
    parsing.Prescan(path, options);
#if __has_include("flang/Support/LangOptions.h")
    Fortran::common::LangOptions langOptions;
    parsing.Parse(llvm::errs(), langOptions);
#else
    parsing.Parse(llvm::errs());
#endif
    const double parseMs = msSince(tParse0);

    auto emitError = [&](const std::string &msg) {
        std::cout << "{\"path\":\"" << jsonEscape(originalPath)
                  << "\",\"error\":\"" << jsonEscape(msg)
                  << "\",\"events\":[]}\n";
        std::cout.flush();
        if (!tempFile.empty()) {
            std::remove(tempFile.c_str());
        }
        return 1;
    };

    if (!parsing.messages().empty() && parsing.messages().AnyFatalError()) {
        parsing.messages().Emit(llvm::errs(), cooked);
        return emitError("flang fatal parse error");
    }
    if (!parsing.parseTree().has_value()) {
        llvm::errs() << "op2-flang-scan: no parse tree produced for " << path << "\n";
        return emitError("no parse tree produced");
    }

    const fp::Program &program = *parsing.parseTree();

    const auto tEmit0 = Clock::now();
    Json json;
    json.beginObject();
    json.key("path");
    json.stringValue(originalPath);
    json.key("events");
    json.beginArray();
    Scanner scanner{json, cooked};
    fp::Walk(program, scanner);
    json.endArray();
    json.endObject();
    const std::string jsonOut = json.str();
    const double walkEmitMs = msSince(tEmit0);

    const auto tWrite0 = Clock::now();
    std::cout << jsonOut << "\n";
    std::cout.flush();
    const double stdoutWriteMs = msSince(tWrite0);

    if (!tempFile.empty()) {
        std::remove(tempFile.c_str());
    }

    // print timing information
    if (emitTiming) {
        const double unitMs = msSince(tUnit);
        std::cerr << "OP2_FLANG_SCAN_TIMING"
                  << " materialize_ms=" << materializeMs
                  << " parse_ms=" << parseMs
                  << " walk_emit_ms=" << walkEmitMs
                  << " stdout_write_ms=" << stdoutWriteMs
                  << " total_ms=" << unitMs
                  << " json_bytes=" << jsonOut.size()
                  << " path=" << originalPath
                  << " session_ms=" << msSince(tSession0)
                  << "\n";
    }
    return 0;
}

/**
 * @brief Run the batched scan protocol.
 *
 * @param includeDirs Extra directories for Fortran INCLUDE resolution.
 * @param emitTiming If true, print OP2_FLANG_SCAN_TIMING lines to stderr.
 * @return 0 unless every unit failed (then 1); 2 on protocol errors.
 * @see scanOneUnit
 */
static int runBatchMode(const std::vector<std::string> &includeDirs, bool emitTiming)
{
    const auto tSession0 = Clock::now();
    std::string magic;
    if (!std::getline(std::cin, magic)) {
        std::cerr << "op2-flang-scan: --batch expected OP2_FLANG_BATCH_V1 header\n";
        return 2;
    }
    if (!magic.empty() && magic.back() == '\r') {
        magic.pop_back();
    }
    if (magic != "OP2_FLANG_BATCH_V1") {
        std::cerr << "op2-flang-scan: bad batch magic: " << magic << "\n";
        return 2;
    }

    int failures = 0;
    int unitIndex = 0;
    while (true) {
        std::string reportedPath;
        if (!std::getline(std::cin, reportedPath)) {
            break; // clean EOF between units
        }
        if (!reportedPath.empty() && reportedPath.back() == '\r') {
            reportedPath.pop_back();
        }
        std::string nbytesLine;
        if (!std::getline(std::cin, nbytesLine)) {
            std::cerr << "op2-flang-scan: truncated batch unit header for "
                      << reportedPath << "\n";
            return 2;
        }
        if (!nbytesLine.empty() && nbytesLine.back() == '\r') {
            nbytesLine.pop_back();
        }
        std::size_t nbytes = 0;
        try {
            nbytes = static_cast<std::size_t>(std::stoull(nbytesLine));
        } catch (...) {
            std::cerr << "op2-flang-scan: bad nbytes in batch stream: "
                      << nbytesLine << "\n";
            return 2;
        }
        std::string body(nbytes, '\0');
        std::size_t got = 0;
        while (got < nbytes) {
            std::cin.read(&body[got], static_cast<std::streamsize>(nbytes - got));
            std::streamsize n = std::cin.gcount();
            if (n <= 0) {
                break;
            }
            got += static_cast<std::size_t>(n);
        }
        if (got != nbytes) {
            std::cerr << "op2-flang-scan: truncated batch body for "
                      << reportedPath << "\n";
            return 2;
        }
        if (scanOneUnit(reportedPath, /*onDiskPath=*/{}, body, includeDirs,
                emitTiming, unitIndex++, tSession0) != 0) {
            ++failures;
        }
    }

    if (emitTiming) {
        std::cerr << "OP2_FLANG_SCAN_BATCH_DONE"
                  << " units=" << unitIndex
                  << " failures=" << failures
                  << " session_ms=" << msSince(tSession0)
                  << "\n";
    }
    // non-zero only if every unit failed (partial success still exits with 0 so
    // Python can apply per-file fparser2 fallback from JSON "error" fields)
    return (unitIndex > 0 && failures == unitIndex) ? 1 : 0;
}

/**
 * @brief Entry point with argument parsing, parse pipeline, and JSON emission.
 *
 * Steps performed:
 *   1. Argument parsing
 *   2. Either batch mode or single-unit mode
 *   3. For each unit: materialise, Prescan+Parse, walk, emit JSON
 *
 * Recognised flags:
 *   --stdin            Read source from stdin even if <path> is given
 *   --batch            Scan many units from an OP2_FLANG_BATCH_V1 stdin stream
 *   --path <reported>  JSON "path" (and directory hint for INCLUDE / temp files)
 *   --timing           Print OP2_FLANG_SCAN_TIMING lines to stderr
 *   -I <dir>           Extra directory for Fortran INCLUDE resolution
 *   <path>             Source file to parse (single-unit mode)
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Process exit status (0 success, 1 parse failure, 2 bad arguments/protocol).
 * @see scanOneUnit
 * @see runBatchMode
 */
int main(int argc, char **argv)
{
    std::string path;
    bool readStdin = false;
    bool batchMode = false;
    bool emitTiming = false;
    std::string originalPath;
    std::vector<std::string> includeDirs;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--stdin") {
            readStdin = true;
        } else if (a == "--batch") {
            batchMode = true;
        } else if (a == "--timing") {
            emitTiming = true;
        } else if (a == "--path" && i + 1 < argc) {
            originalPath = argv[++i];
        } else if (a == "-I" && i + 1 < argc) {
            includeDirs.push_back(argv[++i]);
        } else if (a.rfind("-I", 0) == 0 && a.size() > 2) {
            includeDirs.push_back(a.substr(2));
        } else if (a.size() > 0 && a[0] != '-') {
            path = a;
        } else {
            std::cerr << "op2-flang-scan: unknown argument: " << a << "\n";
            return 2;
        }
    }

    const auto tSession0 = Clock::now();

    if (batchMode) {
        return runBatchMode(includeDirs, emitTiming);
    }

    std::string sourceBytes;
    if (readStdin || path.empty()) {
        sourceBytes = slurpStdin();
    }
    return scanOneUnit(originalPath, path, sourceBytes, includeDirs, emitTiming, /*tempSuffix=*/0, tSession0);
}
