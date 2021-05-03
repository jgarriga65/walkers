
# batch parameters:
# ------------
# g: sgrd instance number
# m: max. flight
# s: max. steps
# d: sensory defective level
# f: gradient fail
# r: number of runs

import sys, getopt
import swalker as sw

class Arg:

    def __init__(self,key,typ,val):
        self.key = key
        self.typ = typ
        self.val = val

def main(argDct):

    b = sw.mpsw(sw.sgLoad(argDct['-g'].val), minf=0.001, maxf=argDct['-m'].val, maxs=argDct['-s'].val, dlvl=argDct['-d'].val, fail=argDct['-f'].val, runs=argDct['-r'].val)
    b.bRun()
    b.saveIt()

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],'g:m:s:d:f:r:')
    except getopt.GetoptError:
        print 'arguments error!'
        sys.exit(2)

    argDct = {}
    for key,typ,val in zip(list('gmsdfr'),list('iiiiii'),[1,10,10000,1,0,1000]):
        argDct['-'+key] = Arg(key,typ,val)

    for opt in opts:
        if argDct[opt[0]].typ == 'i':
            argDct[opt[0]].val = int(opt[1])
        elif argDct[opt[0]].typ == 'f':
            argDct[opt[0]].val = float(opt[1])

   main(argDct)
