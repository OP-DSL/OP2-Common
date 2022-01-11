#include <parmetis.h>

int main() {
    // Dummy function call to make sure linker doesn't do anything strange - 
    // this executable will never actually be run
    int res = ParMETIS_V3_PartKway(nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
				   nullptr);

    return res;
}
