def whitespace_normalization(text:str, remove_nlines=True) -> str:
    """ normalizes whitespace characters & newlines -> single space
    - remove_nlines:
        False -> any combination of whitespaces with \n will be replaced with single \n
        True ->  any combination of whitespaces with \n will be replaced with ' ' """

    def _line_whitespace_normalization(txt: str) -> str:
        if txt: txt = ' '.join(txt.split())
        return txt

    lines = [text] if remove_nlines else text.split('\n')
    lines = [_line_whitespace_normalization(txt=line) for line in lines]
    lines = [l for l in lines if l]  # remove empty lines
    return '\n'.join(lines)