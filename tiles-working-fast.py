import random
import math
# import operator

_maxnumfloats = 20                      # maximum number of variables used in one grid
_maxLongint = 2147483647                # maximum integer
_maxLongintBy4 = _maxLongint // 4       # maximum integer divided by 4
_randomTable = [random.randrange(_maxLongintBy4) for i in xrange(2048)]   #table of random numbers

# The following are temporary variables used by tiles.
_qstate = [0 for i in xrange(_maxnumfloats)]
_base = [0 for i in xrange(_maxnumfloats)]

def startTiles (coordinates, numtilings, floats, ints=[]):
    "Does initial assignments to _coordinates, _base and _qstate for both GetTiles and LoadTiles"
    global _base, _qstate
    numfloats = len(floats)
    i = numfloats + 1                   # starting place for integers
    for v in ints:                      # for each integer variable, store it
        coordinates[i] = v             
        i += 1
    i = 0
    for float in floats:                # for real variables, quantize state to integers
        _base[i] = 0
        _qstate[i] = int(math.floor(float * numtilings))
        i += 1

def fixcoord (coordinates, numtilings, numfloats, j):
    "Fiddles with _coordinates and _base - done once for each tiling"
    global _base, _qstate
    for i in xrange(numfloats):          # for each real variable
        if _qstate[i] >= _base[i]:
            coordinates[i] = _qstate[i] - ((_qstate[i] - _base[i]) % numtilings)
        else:
            coordinates[i] = _qstate[i]+1 + ((_base[i] - _qstate[i] - 1) % numtilings) - numtilings
        _base[i] += 1 + (2*i)
    coordinates[numfloats] = j


def hashUNH (ints, numInts, m, increment=449):
    "Hashing of array of integers into below m, using random table"
    res = 0
    for i in xrange(numInts):
        res += _randomTable[(ints[i] + i*increment) % 2048]
    return res % m


def tiles (numtilings, memctable, floats, ints=[]):
    """Returns list of numtilings tiles corresponding to variables (floats and ints),
        hashed down to mem, using ctable to check for collisions"""

    hashfun = hashUNH

    numfloats = len(floats)
    numcoord = 1 + numfloats + len(ints)
    _coordinates = [0]*numcoord
    startTiles(_coordinates, numtilings, floats, ints)
    tlist = [None] * numtilings
    for j in xrange(numtilings):  # for each tiling
        fixcoord(_coordinates, numtilings, numfloats, j)
        hnum = hashfun(_coordinates, numcoord, memctable)
        tlist[j] = hnum
    return tlist