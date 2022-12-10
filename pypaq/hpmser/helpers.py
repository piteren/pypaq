from typing import List

# returns nice string of floats list
def _str_weights(
        all_w :List[float],
        cut_above=      5,
        float_prec=     4) -> str:
    ws = '['
    if cut_above < 5: cut_above = 5 # cannot be less than 5
    if len(all_w) > cut_above:
        for w in all_w[:3]: ws += f'{w:.{float_prec}f} '
        ws += '.. '
        for w in all_w[-2:]: ws += f'{w:.{float_prec}f} '
    else:
        for w in all_w: ws += f'{w:.{float_prec}f} '
    return f'{ws[:-1]}]'