# coding:utf-8

class Binary_Tree(object):
    def __init__(self, val=None):
        self.root = val
        self.left = None
        self.right = None


def insert_node(tree, val):
    '''插入节点'''
    val_node = Binary_Tree(val)

    if tree.root is None:
        tree.root = val

    else:
        if tree.root < val:
            if tree.right:
                insert_node(tree.right, val)
            else:
                tree.right = val_node

        if tree.root > val:
            if tree.left:
                insert_node(tree.left, val)
            else:
                tree.left = val_node
    return tree



def build_tree(tree, val_list):
    '''根据列表创建二叉树'''
    for val in val_list:
        insert_node(tree, val)
    return tree


def pre_order_print(tree, pre_list = []):
    '''前序遍历'''
    if tree is not None:
        pre_list.append(tree.root)
        pre_order_print(tree.left, pre_list)
        pre_order_print(tree.right, pre_list)
    return pre_list


def in_order_print(tree, in_list = []):
    '''中序遍历'''
    if tree is not None:
        in_order_print(tree.left, in_list)
        in_list.append(tree.root)
        in_order_print(tree.right, in_list)
    return in_list


def post_order_print(tree, post_list = []):
    '''后序遍历'''
    if tree is not None:
        post_order_print(tree.left, post_list)
        post_order_print(tree.right, post_list)
        post_list.append(tree.root)
    return post_list


def find_min(tree):
    '''查找最小值'''
    while tree.left is not None:
        tree = tree.left
    return tree.root


def find_max(tree):
    '''查找最大值'''
    while tree.right is not None:
        tree = tree.right
    return tree.root


def find_val(tree, val):
    '''查找特定值'''
    find = False
    while tree is not None:

        if tree.root > val:
            tree = tree.left

        elif tree.root < val:
            tree = tree.right

        elif tree.root == val:
            find = True
            print "finding %s successfully" % val
            break
    if find is False:
        print "can not find %s" % val

    return find



if __name__ == '__main__':
    bt = Binary_Tree()
    bt = build_tree(bt, [4,6,5,9,1,12,7,10])
    find_val(bt, 18)
