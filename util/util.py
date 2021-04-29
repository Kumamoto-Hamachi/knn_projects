def argmax_for_dict(d):
    max_value = 0
    for key, value in d.items():
        if value > max_value:
            max_value = value
            key_with_max_value = key
    return key_with_max_value, max_value

def argmax_for_list(l):  #TODO
    raise RuntimeError('this func not yet implemented')
