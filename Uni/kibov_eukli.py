def kibovitett_euklidesz(a, b):
    regi_r, r = a, b
    regi_s, s = 1, 0
    regi_t, t = 0, 1

    while r != 0:
        hanyados = regi_r // r
        regi_r, r = r, regi_r - hanyados * r
        regi_s, s = s, regi_s - hanyados * s
        regi_t, t = t, regi_t - hanyados * t

    return (regi_r, regi_s, regi_t)
