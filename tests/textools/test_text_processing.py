from pypaq.textools.text_processing import whitespace_normalization


def test_whitespace_normalization():
    for t,r in [
        ('', ''),
        (' ', ''),
        ('\n', ''),
        ('a\na', 'a a'),
        ('a\n  a', 'a a'),
        ('a\n  \n\na', 'a a'),
    ]:
        rp = whitespace_normalization(t)
        print(f'>{rp}<->{r}<')
        assert rp == r
