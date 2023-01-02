# normalizes whitespace characters (newlines also) in given string to single space
def whitespace_normalization(
        text: str,
        remove_nlines=  True) -> str:  # for False leaves single (space separated) '\n' between lines, for True replaces '\n' with space

    def _line_whitespace_normalization(txt: str) -> str:
        if txt: txt = ' '.join(txt.split())
        return txt

    lines = [text] if remove_nlines else text.split('\n')
    lines = [_line_whitespace_normalization(txt=line) for line in lines]
    lines = [l for l in lines if l]  # remove empty lines
    return '\n'.join(lines)