import itertools
import sys

corpus = [('green house','casa verde'),('the house','la casa')]
# Print corpus:
vocab1 = []
vocab2 = []
print "Sentence pairs"
for i in range(len(corpus)):
    tup = corpus[i]
    print i,
    print '%s\t%s' % tup 
    vocab1 += tup[0].split()
    vocab2 += tup[1].split()

# Print Vocabulary
vocab1 = list(set(vocab1))
vocab2 = list(set(vocab2))
print
print "Vocabulary"
print "Source Language:",
print vocab1
print "Target Language:",
print vocab2

print
print "EM initliazation"
prob = {}
for w in vocab1:
    for v in vocab2:
        prob[(w,v)] = 1. / len(vocab2)
        print "P(%s|%s) = %.2f\t" % (v,w,prob[(w,v)]),
    print

def E_step(prob):
    print "E_step"
    def compute_align(a,sent_pair):
        print sent_pair
        print "\t Alignment:",
        p = 1.
        s = sent_pair[0].split()
        t = sent_pair[1].split()
        for i in range(len(a)):
            w = s[i]
            v = t[a[i]]
            print (w,v),
            p = p * prob[(w,v)]
        print 
        print "\t p(a,f|e): %.2f" % p
        return p
        
    new_prob = {}
    for w in vocab1:
        for v in vocab2:
            new_prob[(w,v)] = 0.

    for i in range(len(corpus)):
        print "Sentence Pair",i
        sent_pair = corpus[i]
        sent_l = len(sent_pair)    
        total_i = []
        for a in itertools.permutations(range(sent_l)):
            total_i.append(compute_align(a, sent_pair))
        #normalize
        #print "\tp(a,f|e):",total_i
        total_i_sum = sum(total_i)
        total_i = [t / total_i_sum for t in total_i]
        print "\n\t Normalizing"
        print "\t p(a|e,f):",total_i
        print

        s = sent_pair[0].split()
        t = sent_pair[1].split()
        cnt = 0
        for a in itertools.permutations(range(sent_l)):
            for j in range(len(a)):
                w = s[j]
                v = t[a[j]]
                new_prob[(w,v)] += total_i[cnt]
            cnt += 1
          
    for w in vocab1:
        total_w = 0.
        for v in vocab2:
            total_w += new_prob[(w,v)]
            print "P(%s|%s) = %.2f\t" % (v,w,new_prob[(w,v)]),
        print "total(%s) = %2.f" % (w,total_w)

    return new_prob

def M_step(prob):
    print "M_step"
    for w in vocab1:
        total_w = sum([prob[w,v] for v in vocab2])
        for v in vocab2:
            prob[(w,v)] = prob[(w,v)] / total_w
            print "P(%s|%s) = %.2f\t" % (v,w,prob[(w,v)]),
        print
    return prob

iter_num = sys.argv[1]
for i in range(int(iter_num)):
    print 'iter',i
    prob = E_step(prob)
    prob = M_step(prob)
