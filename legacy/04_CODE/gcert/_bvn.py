"""Vectorised bivariate-normal helpers (Genz 2004 BVNU and partial-moment closed forms)."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

SQ2PI = math.sqrt(2.0 * math.pi)


def npdf(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


_GX = {6: (np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970]),
           np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])),
       12: (np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                      0.5873179542866171, 0.3678314989981802, 0.1252334085114692]),
            np.array([.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                      0.2031674267230659, 0.2334925365383547, 0.2491470458134029])),
       20: (np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259, 0.8391169718222188,
                      0.7463319064601508, 0.6360536807265150, 0.5108670019508271, 0.3737060887154196,
                      0.2277858511416451, 0.07652652113349733]),
            np.array([.01761400713915212, .04060142980038694, .06267204833410906, .08327674157670475,
                      0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183821,
                      0.1491729864726037, 0.1527533871307259]))}


def bvn_upper(h, k, r):
    """P(X > h, Y > k), standard bivariate normal with correlation r (vectorised Genz BVNU)."""
    h, k, r = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(r, float))
    shape = h.shape
    h, k = h.ravel().astype(float).copy(), k.ravel().astype(float).copy()
    r = np.clip(r.ravel().astype(float), -1.0, 1.0)
    out = np.zeros(h.size)
    tp = 2.0 * np.pi
    hk = h * k
    ar = np.abs(r)
    for lo, hi, ng in ((0.0, 0.3, 6), (0.3, 0.75, 12), (0.75, 0.925, 20)):
        m = (ar >= lo) & (ar < hi)
        if not np.any(m):
            continue
        x, w = _GX[ng]
        X = np.concatenate([1 - x, 1 + x])
        Wt = np.concatenate([w, w])
        hs = 0.5 * (h[m] ** 2 + k[m] ** 2)
        asr = 0.5 * np.arcsin(r[m])
        sn = np.sin(asr[:, None] * X[None, :])
        bvn = np.exp((sn * hk[m][:, None] - hs[:, None]) / (1 - sn ** 2)) @ Wt
        out[m] = bvn * asr / tp + ndtr(-h[m]) * ndtr(-k[m])
    m = ar >= 0.925
    if np.any(m):
        x, w = _GX[20]
        X = np.concatenate([1 - x, 1 + x])
        Wt = np.concatenate([w, w])
        hh, kk, rr = h[m], k[m].copy(), r[m]
        hkk = hh * kk
        neg = rr < 0
        kk[neg] = -kk[neg]
        hkk[neg] = -hkk[neg]
        bvn = np.zeros(hh.size)
        inner = np.abs(rr) < 1
        if np.any(inner):
            hi_, ki_, hk_ = hh[inner], kk[inner], hkk[inner]
            as_ = 1 - rr[inner] ** 2
            a = np.sqrt(as_)
            bs = (hi_ - ki_) ** 2
            asr = -(bs / as_ + hk_) / 2
            c = (4 - hk_) / 8
            d = (12 - hk_) / 80
            b1 = np.where(asr > -100,
                          a * np.exp(np.maximum(asr, -700)) * (1 - c * (bs - as_) * (1 - d * bs) / 3 + c * d * as_ ** 2),
                          0.0)
            b = np.sqrt(bs)
            sp = math.sqrt(tp) * ndtr(-b / a)
            b1 = np.where(hk_ > -100, b1 - np.exp(-np.maximum(hk_, -700) / 2) * sp * b * (1 - c * bs * (1 - d * bs) / 3), b1)
            a2 = a / 2
            xs = (a2[:, None] * X[None, :]) ** 2
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                asr2 = -(bs[:, None] / xs + hk_[:, None]) / 2
                spp = 1 + c[:, None] * xs * (1 + 5.0 * d[:, None] * xs)   # Genz: D = (12 - hk)/16 = 5 d
                rs = np.sqrt(1 - xs)
                ep = np.exp(-(hk_[:, None] / 2) * xs / (1 + rs) ** 2) / rs
                term = np.where(asr2 > -100, np.exp(np.maximum(asr2, -700)) * (spp - ep), 0.0)
            bvn[inner] = (a2 * (term @ Wt) - b1) / tp
        res = np.empty(hh.size)
        pos = rr > 0
        res[pos] = bvn[pos] + ndtr(-np.maximum(hh[pos], kk[pos]))
        ngm = ~pos
        hge = ngm & (hh >= kk)
        res[hge] = -bvn[hge]
        oth = ngm & (hh < kk)
        Lh = np.where(hh[oth] < 0, ndtr(kk[oth]) - ndtr(hh[oth]), ndtr(-hh[oth]) - ndtr(-kk[oth]))
        res[oth] = Lh - bvn[oth]
        out[m] = res
    return np.clip(out, 0.0, 1.0).reshape(shape)


def bvn_upper_integral(h, k, r, n=64):
    """Slow independent reference: int_h^inf phi(x) Phibar((k - r x)/q) dx, graded at x = k/r."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    h, k, r = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(r, float))
    out = np.empty(h.shape)
    for idx, (hh, kk, rr) in enumerate(zip(h.ravel(), k.ravel(), r.ravel())):
        if rr >= 1.0 - 1e-15:
            val = float(ndtr(-max(hh, kk)))
        elif rr <= -1.0 + 1e-15:
            val = float(max(ndtr(-hh) - ndtr(kk), 0.0))
        else:
            sq = math.sqrt(1.0 - rr * rr)
            lo, hi = hh, max(hh, 0.0) + 12.0
            cuts = [lo, hi]
            if abs(rr) > 1e-14:
                zc = kk / rr
                if lo < zc < hi:
                    cuts.append(zc)
                    width = max(sq / abs(rr), 1e-15)
                    cuts += [zc + sg * width * 4.0 ** (j - 4) for j in range(1, 10) for sg in (-1, 1)
                             if lo < zc + sg * width * 4.0 ** (j - 4) < hi]
            cuts = np.unique(np.asarray(cuts))
            acc = 0.0
            for a, b in zip(cuts[:-1], cuts[1:]):
                z = 0.5 * (b - a) * (xg + 1) + a
                acc += float(np.sum(0.5 * (b - a) * wg * npdf(z) * ndtr(-(kk - rr * z) / sq)))
            val = acc
        out.flat[idx] = val
    return out


def std_call_product(al, be, r):
    """E[(X - al)_+ (Y - be)_+], standard bivariate normal, correlation r (vectorised).

    |r| < 1:  E = E[XY 1_A] - be E[X 1_A] - al E[Y 1_A] + al be P(A),  A = {X > al, Y > be},
      P(A) = BVNU(al, be, r),
      E[X 1_A] = phi(al) Phibar((be - r al)/q) + r phi(be) Phibar((al - r be)/q),
      E[XY 1_A] = r P(A) + r al phi(al) Phibar((be - r al)/q) + r be phi(be) Phibar((al - r be)/q)
                  + q phi(al) phi((be - r al)/q),   q = sqrt(1 - r^2).
    r = 1:  (1 + al be) Phibar(m) + (m - al - be) phi(m),  m = max(al, be).
    r = -1: int_al^{-be} (x - al)(-x - be) phi(x) dx in closed form.
    """
    al, be, r = (np.asarray(v, float) for v in np.broadcast_arrays(al, be, r))
    out = np.empty(al.shape)
    up = r >= 1.0 - 1e-13
    dn = r <= -1.0 + 1e-13
    it = ~(up | dn)
    if np.any(up):
        m = np.maximum(al[up], be[up])
        out[up] = (1 + al[up] * be[up]) * ndtr(-m) + (m - al[up] - be[up]) * npdf(m)
    if np.any(dn):
        A, B = al[dn], be[dn]
        lo, hi = A, -B

        def F(t):
            # antiderivative of (-x^2 + (A - B) x + A B) phi(x)
            return -(ndtr(t) - t * npdf(t)) - (A - B) * npdf(t) + A * B * ndtr(t)
        out[dn] = np.where(hi > lo, F(hi) - F(lo), 0.0)
    if np.any(it):
        A, B, R = al[it], be[it], r[it]
        q = np.sqrt(1.0 - R * R)
        P2 = bvn_upper(A, B, R)
        A1 = npdf(A) * ndtr(-(B - R * A) / q)
        B1 = npdf(B) * ndtr(-(A - R * B) / q)
        EX = A1 + R * B1
        EY = B1 + R * A1
        EXY = R * P2 + R * A * A1 + R * B * B1 + q * npdf(A) * npdf((B - R * A) / q)
        out[it] = EXY - B * EX - A * EY + A * B * P2
    return out


def bivariate_orthant(a, b, sa, sb, x):
    """P(U > a, V > b), Var U = sa^2, Var V = sb^2, Cov = x (vectorised)."""
    a, b, sa, sb, x = (np.asarray(v, float) for v in np.broadcast_arrays(a, b, sa, sb, x))
    out = np.empty(a.shape)
    both = (sa > 0) & (sb > 0)
    if np.any(both):
        r = np.clip(x[both] / (sa[both] * sb[both]), -1.0, 1.0)
        out[both] = bvn_upper(a[both] / sa[both], b[both] / sb[both], r)
    oa = (sa <= 0) & (sb > 0)
    if np.any(oa):
        out[oa] = (0 > a[oa]) * ndtr(-b[oa] / sb[oa])
    ob = (sb <= 0) & (sa > 0)
    if np.any(ob):
        out[ob] = (0 > b[ob]) * ndtr(-a[ob] / sa[ob])
    nn = (sa <= 0) & (sb <= 0)
    out[nn] = ((0 > a[nn]) & (0 > b[nn])).astype(float)
    return out


def bivariate_call_product(a, b, sa, sb, x):
    """E[(U - a)_+ (V - b)_+] for a centred bivariate normal (vectorised closed form)."""
    a, b, sa, sb, x = (np.asarray(v, float) for v in np.broadcast_arrays(a, b, sa, sb, x))
    out = np.empty(a.shape)
    both = (sa > 0) & (sb > 0)
    if np.any(both):
        r = np.clip(x[both] / (sa[both] * sb[both]), -1.0, 1.0)
        out[both] = sa[both] * sb[both] * std_call_product(a[both] / sa[both], b[both] / sb[both], r)
    oa = (sa <= 0) & (sb > 0)
    if np.any(oa):
        al = b[oa] / sb[oa]
        out[oa] = np.maximum(-a[oa], 0.0) * sb[oa] * (npdf(al) - al * ndtr(-al))
    ob = (sb <= 0) & (sa > 0)
    if np.any(ob):
        al = a[ob] / sa[ob]
        out[ob] = np.maximum(-b[ob], 0.0) * sa[ob] * (npdf(al) - al * ndtr(-al))
    nn = (sa <= 0) & (sb <= 0)
    out[nn] = np.maximum(-a[nn], 0.0) * np.maximum(-b[nn], 0.0)
    return out
