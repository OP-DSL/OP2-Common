from subprocess import check_output, CalledProcessError
from docutils import nodes

def run_cmd(cmd):
    try:
        return check_output(cmd).strip().decode('utf-8')
    except CalledProcessError:
        return None

def get_github_rev():
    path = run_cmd(['git', 'rev-parse', '--short', 'HEAD'])
    tag = run_cmd(['git', 'describe', '--exact-match'])

    if tag:
        return tag

    return path

def setup(app):
    baseurl = 'https://github.com/OP-DSL/OP2-Common'
    rev = get_github_rev()

    app.add_role('gh-blob', autolink('{}/blob/{}/%s'.format(baseurl, rev)))
    app.add_role('gh-tree', autolink('{}/tree/{}/%s'.format(baseurl, rev)))

def autolink(pattern):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = pattern % (text,)
        node = nodes.reference(rawtext, text, refuri=url, **options)
        return [node], []

    return role

