package com.datastructures.Tree.treeTraverse;



import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;
import java.util.Stack;

public class BinaryTree
{
    static class Node
    {
        int data;
        Node left,right;
        public Node(int data)
        {
            this.data=data;
            left=null;
            right=null;
        }
    }

    static Scanner sc = new Scanner(System.in);

    public static void main(String[] args) {

        Node root = createTree();
//        inOrder(root);
//        System.out.println();
//        preOrder(root);
//        System.out.println();
//        postOrder(root);
//        System.out.println();
//        levelOrder(root);
//        System.out.println();
//        delete(root,3);
//        height(root);
//        System.out.println();
        reverseLevelOrder(root);
    }

    static Node createTree()
    {

        System.out.println("Enter data: ");
        int data = sc.nextInt();

        if(data == -1) return null;

        Node root = new Node(data);

        System.out.println("Enter left for " + data);
        root.left = createTree();

        System.out.println("Enter right for "+ data);
        root.right = createTree();

        return root;
    }

    static void inOrder(Node root)
    {
        if(root == null) return;

        inOrder(root.left);
        System.out.print(root.data+" ");
        inOrder(root.right);
    }

    static void preOrder(Node root)
    {
        if(root == null) return;
        System.out.print(root.data+" ");
        preOrder(root.left);
        preOrder(root.right);
    }

    static void postOrder(Node root)
    {
        if(root == null) return;

        postOrder(root.left);
        postOrder(root.right);
        System.out.print(root.data+" ");
    }

    static void inOrderIterative(Node root)
    {
        if(root==null) return;

        Stack<Node> stack = new Stack<>();
        Node curr = root;
        while(curr!=null || stack.size()>0)
        {
            while(curr!=null)
            {
                stack.push(curr);
                curr=curr.left;
            }
            curr=stack.pop();
            System.out.print(curr.data);
            curr=curr.right;
        }
    }

    static void preOrderIterative(Node root)
    {
        if (root == null)
        {
            return;
        }
        Stack<Node> st = new Stack<>();
        Node curr = root;

        while (curr != null || !st.isEmpty())
        {
            while (curr != null)
            {
                System.out.print(curr.data + " ");

                if (curr.right != null)
                    st.push(curr.right);

                curr = curr.left;
            }
            if (!st.isEmpty())
            {
                curr = st.pop();
            }
        }
    }


    static void postOrderIterative(Node root)
    {
        Stack<Node> stack = new Stack<>();
        while(true) {
            while(root != null) {
                stack.push(root);
                stack.push(root);
                root = root.left;
            }

            // Check for empty stack
            if(stack.empty()) return;
            root = stack.pop();

            if(!stack.empty() && stack.peek() == root) root = root.right;

            else {

                System.out.print(root.data + " "); root = null;
            }
        }
    }

    static void levelOrder(Node node)
    {

        if(node == null)
            return ;

        Queue<Node> q = new LinkedList<>();
        q.add(node);
        while(!q.isEmpty())
        {

            node = q.peek();
            System.out.print(node.data+" ");
            q.poll();

            if(node.left != null)
                q.add(node.left);

            if(node.right != null)
                q.add(node.right);

        }
    }

    static void insertLevelOrder(Node temp, int key)
    {
        Queue<Node> q = new LinkedList<>();
        q.add(temp);

        // Do level order traversal until we find
        // an empty place.
        while (!q.isEmpty())
        {
            temp = q.peek();
            q.remove();

            if (temp.left == null)
            {
                temp.left = new Node(key);
                break;
            }
            else
                q.add(temp.left);

            if (temp.right == null)
            {
                temp.right = new Node(key);
                break;
            }
            else
                q.add(temp.right);
        }
    }



    // Function to delete given element in binary tree
    static void delete(Node root, int key)
    {
        if (root == null)
            return;

        if (root.left == null &&
                root.right == null)
        {
            if (root.data == key)
                return;
            else
                return;
        }

        Queue<Node> q = new LinkedList<>();
        q.add(root);
        Node temp = null, keyNode = null;

        // Do level order traversal until
        // we find key and last node.
        while (!q.isEmpty())
        {
            temp = q.peek();
            q.remove();

            if (temp.data == key)
                keyNode = temp;

            if (temp.left != null)
                q.add(temp.left);

            if (temp.right != null)
                q.add(temp.right);
        }

        if (keyNode != null)
        {
            int x = temp.data;
            deleteDeepest(root, temp);
            keyNode.data = x;
        }
    }

    static void deleteDeepest(Node root, Node delNode)
    {
        Queue<Node> q = new LinkedList<Node>();
        q.add(root);

        Node temp = null;

        // Do level order traversal until last node
        while (!q.isEmpty())
        {
            temp = q.peek();
            q.remove();

            if (temp == delNode)
            {
                temp = null;
                return;

            }
            if (temp.right!=null)
            {
                if (temp.right == delNode)
                {
                    temp.right = null;
                    return;
                }
                else
                    q.add(temp.right);
            }

            if (temp.left != null)
            {
                if (temp.left == delNode)
                {
                    temp.left = null;
                    return;
                }
                else
                    q.add(temp.left);
            }
        }
    }


    //height of tree
    static int height(Node root)
    {
        if(root==null)
            return 0;
        else
        {
            int lChild = height(root.left);
            int rChild = height(root.right);
            if(lChild>rChild)
                return lChild+1;
            else
                return rChild+1;
        }
    }


    //reverse order traversal

    static void reverseLevelOrder(Node node)
    {
        int h = height(node);
        int i;
        for (i = h; i >= 1; i--)
        //THE ONLY LINE DIFFERENT FROM NORMAL LEVEL ORDER
        {
            printGivenLevel(node, i);
        }
    }

    /* Print nodes at a given level */
    static void printGivenLevel(Node node, int level)
    {
        if (node == null)
            return;
        if (level == 1)
        {
            System.out.print(node.data + " ");
        }
        else if (level > 1)
        {
            printGivenLevel(node.left, level - 1);
            printGivenLevel(node.right, level - 1);
        }
    }


    //reverse order using stack and queue

    static void reverseLevel(Node node) {
        Stack<Node> S = new Stack<>();
        Queue<Node> Q = new LinkedList<>();
        Q.add(node);

        // Do something like normal level order traversal order.Following
        // are the differences with normal level order traversal
        // 1) Instead of printing a node, we push the node to stack
        // 2) Right subtree is visited before left subtree
        while (!Q.isEmpty()) {
            /* Dequeue node and make it root */
            node = Q.peek();
            Q.remove();
            S.push(node);

            /* Enqueue right child */
            if (node.right != null)
                // NOTE: RIGHT CHILD IS ENQUEUED BEFORE LEFT
                Q.add(node.right);

            /* Enqueue left child */
            if (node.left != null)
                Q.add(node.left);
        }

        // Now pop all items from stack one by one and print them
        while (!S.empty()) {
            node = S.peek();
            System.out.print(node.data + " ");
            S.pop();
        }
    }


    //search node in tree

    static boolean ifNodeExists( Node node, int key)
    {
        if (node == null)
            return false;

        if (node.data == key)
            return true;

        // then recur on left subtree /
        boolean res1 = ifNodeExists(node.left, key);
        if(res1) return true; // node found, no need to look further

        // node is not found in left, so recur on right subtree /
        boolean res2 = ifNodeExists(node.right, key);

        return res2;
    }


    //diameter of tree

    static int diameter(Node root)
    {
        if (root == null)
            return 0;

        int lheight = heightT(root.left);
        int rheight = heightT(root.right);

        int ldiameter = diameter(root.left);
        int rdiameter = diameter(root.right);

        return Math.max(lheight + rheight + 1,
                Math.max(ldiameter, rdiameter));

    }
    static int heightT(Node node)
    {
        if (node == null)
            return 0;

        return (1 + Math.max(height(node.left), height(node.right)));
    }


    //mirror of binary tree

    void mirror(Node node)
    {
        if (node == null)
            return;

        mirror(node.left);
        mirror(node.right);

        Node temp = node.left;
        node.left = node.right;
        node.right = temp;
    }


    //check tree is balanced or not

    boolean isBalanced(Node root)
    {
        // Your code here
        if(root==null)
        {
            return true;
        }

        int l = height(root.left);
        int r = height(root.right);

        if(Math.abs(l-r)<=1 && isBalanced(root.left) && isBalanced(root.right))
        {
            return true;
        }
        return false;
    }


    //find max of node

    static int findMax(Node node)
    {
        if (node == null)
            return Integer.MIN_VALUE;

        int res = node.data;
        int lres = findMax(node.left);
        int rres = findMax(node.right);

        if (lres > res)
            res = lres;
        if (rres > res)
            res = rres;
        return res;
    }


    //find min of node

    static int findMin(Node node)
    {
        if (node == null)
            return Integer.MAX_VALUE;

        int res = node.data;
        int lres = findMin(node.left);
        int rres = findMin(node.right);

        if (lres < res)
            res = lres;
        if (rres < res)
            res = rres;
        return res;
    }




}
