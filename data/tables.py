from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

if __name__ == "__main__":
    valids = set()
    for frsqr in range(64):
        for tosqr in range(64):
            fr, ff = frsqr % 8, frsqr >> 3
            tr, tf = tosqr % 8, tosqr >> 3
            dr, df = abs(tr - fr), abs(tf - ff)
            straight = tr == fr or tf == ff
            diag = dr == df
            horse = (dr == 2 and df == 1) or (dr == 1 and df == 2)
            if (straight or diag or horse) and frsqr != tosqr:
                valids.add((frsqr,tosqr,0))
    def addprom(frsqr, tosqr):
        valids.add((frsqr, tosqr, 1))
        valids.add((frsqr, tosqr, 2))
        valids.add((frsqr, tosqr, 3))
    for frsqr in range(48, 56):
        f = frsqr % 8
        addprom(frsqr, frsqr + 8)
        if f > 0: addprom(frsqr, frsqr + 7)
        if f < 7: addprom(frsqr, frsqr + 9)

    move_map = Tensor.empty(1858-66, device="DISK:tensors/move_map.bin", dtype=dtypes.int32)
    non_promos = [fr*64 + to_ for fr, to_, p in valids if not p]
    move_map.assign(Tensor(sorted(non_promos)))
