

for l in open('cpis_kinase_5000posneg_test.trial_1.txt'):
    seq = l.strip('()\n').split(',')
    chem,prot,stat = seq[0].replace('chemID=','').strip(' '), seq[1].replace('protID=','').strip(' '), seq[2].replace('status=','').strip(' ')
    assert stat in ['interaction', 'non-interaction'], stat
    stat = 1 if stat == 'interaction' else 0 
    print '%s--%s,%d' % (chem,prot,stat)

