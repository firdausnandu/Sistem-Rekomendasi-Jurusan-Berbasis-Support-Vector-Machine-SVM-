from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)
@register.filter
def replace(value, args):
    """
    args harus string "old,new"
    contoh: {{ somevar|replace:"nilai_, " }}
    """
    old, new = args.split(',')
    return value.replace(old, new)