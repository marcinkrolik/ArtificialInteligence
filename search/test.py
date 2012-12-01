def myLog(x,b):    
    
    if type(x) != "<type 'int'>" or type(b) != "<type 'int'>": 
        check = False
    elif type(x) == "<type 'int'>" and type(b) == "<type 'int'>":
        check = True
        
    if check and x > 0 and b >= 2:
        if x == 1:
            return 0
        index = 0
        while b >= x:
            b = b/x
            index += 1
            print x, b, index
        return index

def myLog2(x,b):
    result = 0
    while b ** result < x:
        result = result + 1
    if b ** result != x:
        return b-1
    else:
        return result

def myLog3(x, b):
    b1 = b
    i = 1
    if x < b:
        return 0
    while b1*b <= x:
        if b == 0:
            return 0
        i+=1
        b1*= b
    return i

print myLog3(0, 0)


i = myLog(-10.0,2)

def laceStrings(s1, s2):
    
    l1 = len(s1)
    l2 = len(s2)
    shared = ''
    tail = ''
    index = l1
    
    if l1 > l2:
        tail += s1[l2:]
        print tail
        index = l2
    elif l1 < l2:
        tail += s2[l1:]
        print tail
        index = l1 
    
    i = 0
    for char in s1[:index]:
        shared += char
        shared += s2[i]
        i += 1
    shared += tail
    return shared

s1 = 'abc def'
s2 = '12345'

#print laceStrings(s1, s2)


def laceStrings2(s1, s2):
    s = 0
    s3 = ''
    if len(s1) > len(s2):
        for s in range(len(s2)):
            s3 += s1[s]
            s3 += s2[s]
        s3 += s1[s+1:]
    
    elif len(s1) > len(s2):
        for s in range(len(s1)):
            s3 += s1[s]
            s3 += s2[s]
        s3 += s2[:]
    
    else:
        for s in range(len(s1)):
            s3 += s1[s]
            s3 += s2[s]
    
    return s3
    
#print laceStrings2(s1, s2)


def laceStringsRecur(s1, s2):
    """
    s1 and s2 are strings.

    Returns a new str with elements of s1 and s2 interlaced,
    beginning with s1. If strings are not of same length, 
    then the extra elements should appear at the end.
    """
    def helpLaceStrings(s1, s2, out):
        if s1 == '':
            return out + s2
        if s2 == '':
            return out + s1
        else:
            return helpLaceStrings(s1[1:], s2[1:], out + s1[0] + s2[0])
    return helpLaceStrings(s1, s2, '')

#print "recur", laceStringsRecur(s1, s2)

def ll(s1, s2):
    result = ''
    while True:
        if s1 == '':
            result = result + s2
            break
        elif s2 == '':
            result = result + s1
            break
        else:
            result = result + s1[0] + s2[0]
            s1 = s1[1:]
            s2 = s2[2:]
    return result

#print "ll ", ll(s1, s2)

