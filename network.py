import pybel

def json(mol):
    import re
    factor = 18
    atom = bond = ''
    for line in mol.write('mol').split('\n'):
        seq = re.split('\s+', line.strip())
        if len(seq) == 16:
            x,y,l = float(seq[0]) * factor,float(seq[1]) * factor,seq[3]
            if l == 'C':
                atom += '{x:%.2f,y:%.2f},' % (x,y)
            else:
                atom += '{x:%.2f,y:%.2f,l:"%s"},' % (x,y,l)
        if len(seq) == 7 and seq[0] != 'M':
            b,e,o = int(seq[0]),int(seq[1]),int(seq[2])
            b,e = b-1,e-1
            if o > 1:
                bond += '{b:%d,e:%d,o:%d},' % (b,e,o)
            else:
                bond += '{b:%d,e:%d},' % (b,e)
    return '{a:[%s],b:[%s]}' % (atom.strip(','),bond.strip(','))
