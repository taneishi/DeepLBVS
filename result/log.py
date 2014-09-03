import sys
from pylab import *

def main():
    if len(sys.argv) < 2:
        sys.exit('%s [filename]' % sys.argv[0])
    for filename in sys.argv[1:]:
        x = []
        y = []
        for l in open(filename):
            l = l.strip(' \n')
            if not l.startswith('epoch'): continue
            l = l.replace('epoch ','').replace(' minibatch ','').replace(' validation error ','').replace(' %','')
            seq = l.split(',')
            if seq[2].startswith(' test error'): continue

            x.append(seq[0])
            y.append(seq[2])

        figure(figsize=(8,6),dpi=80)
        ylim(0,100)
        ylabel('validation error(%)')
        xlim(0,1000)
        xlabel('epochs(times)')
        title(filename.replace('.log',''))

        plot(x, y) 
        #show()

        savefig(filename.replace('log','png'))

main()
