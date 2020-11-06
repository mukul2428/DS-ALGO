package com.datastructures.Tree.BST;

import com.datastructures.Tree.Dfstraverse.DFSUsingBST;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class BST
{
    static class Node
    {
        Node left,right;
        int data;
        Node(int data)
        {
            this.data=data;
            left=right=null;
        }
    }

    public static int getHeight(Node root)
    {
        //Write your code here
            int heightLeft = 0;
            int heightRight = 0;

            if (root.left != null)
            {
                heightLeft = getHeight(root.left) + 1;
            }
            if (root.right != null)
            {
                heightRight = getHeight(root.right) + 1;
            }

            return (Math.max(heightLeft, heightRight));

    }

    public static Node insert(Node root,int data)
    {
        if(root==null)
        {
            return new Node(data);
        }
        else {
            Node cur;
            if(data<=root.data)
            {
                cur=insert(root.left,data);
                root.left=cur;
            }
            else {
                cur=insert(root.right,data);
                root.right=cur;
            }
            return root;
        }
    }

    public static Node search(Node root, int key)
    {
        // Base Cases: root is null or key is present at root
        if (root==null || root.data==key)
            return root;

        // val is greater than root's key
        if (root.data > key)
            return search(root.left, key);


        // val is less than root's key
        return search(root.right, key);
    }

    static void inorder(Node root)
    {
        if(root==null)
            return;
        inorder(root.left);
        System.out.print(root.data+" ");
        inorder(root.right);
    }

    static void levelOrder(Node root)
    {
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);

        while(!queue.isEmpty())
        {
            Node current = queue.remove();
            System.out.print(current.data+" ");
            if (current.left!=null) queue.add(current.left);
            if (current.right!=null) queue.add(current.right);
        }
    }


    // tree is bst or not

    boolean isBST(Node root)
    {
        return isBSTUtil(root, Integer.MIN_VALUE,
                Integer.MAX_VALUE);
    }

    boolean isBSTUtil(Node node, int min, int max)
    {

        if (node == null)
            return true;

        if (node.data < min || node.data > max)
            return false;

        return (isBSTUtil(node.left, min, node.data-1) &&
                isBSTUtil(node.right, node.data+1, max));
    }


    public static void main(String args[])
    {
        Scanner sc=new Scanner(System.in);
        int T=sc.nextInt();
        Node root=null;
        while(T-->0)
        {
            int data=sc.nextInt();
            root=insert(root,data);
        }
        Node s = search(root,5);
        System.out.println(s.data);
        inorder(root);
        int height=getHeight(root);
        System.out.println("\n"+height);
    }
}
