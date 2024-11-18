def xor(x1, x2):
    return (x1 or x2) and not (x1 and x2)


def bitwise_xor(x1, x2):
    return (x1 | x2) and not (x1 & x2)


assert not xor(True, True)
assert xor(True, False)
assert not xor(False, False)

assert not bitwise_xor(1, 1)
assert bitwise_xor(1, 0)
assert not bitwise_xor(0, 0)