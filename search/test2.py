def sort2(lst):
    for iteration in range(len(lst)):
        minIndex = iteration
        minValue = lst[iteration]
        for j in range(iteration+1, len(lst)):
            if lst[j] < minValue:
                minIndex = j
                minValue = lst[j]
        temp = lst[iteration]
        lst[iteration] = minValue
        lst[minIndex] = temp
        print lst
        print l
        aa = raw_input()
        # the questions below refer to this point, assuming we set L = lst[:] here
    return lst

def sort3(lst):
    out = []
    for iteration in range(0,len(lst)):
        new = lst[iteration]
        inserted = False
        for j in range(len(out)):
            if new < out[j]:
                out.insert(j, new)
                inserted = True
                break
            print l, out
        if not inserted:
            out.append(new)
        # the questions below refer to this point, assuming we set L = out[:] here
    return out

def sort4(lst):
    def unite(l1, l2):
        if len(l1) == 0:
            return l2
        elif len(l2) == 0:
            return l1
        elif l1[0] < l2[0]:
            return [l1[0]] + unite(l1[1:], l2)
        else:
            return [l2[0]] + unite(l1, l2[1:])

    if len(lst) == 0 or len(lst) == 1:
        return lst
    else:
        front = sort4(lst[:len(lst)/2])
        back = sort4(lst[len(lst)/2:])
        # the questions below refer to this point, assuming we set L = lst[:] here
        print front, back
        return unite(front, back)


l =  [4,1,3,55,2,8,11,6,61,7]
h =  [4,1,3,55,2,8,11,6,61,7]

print sort4(h)

