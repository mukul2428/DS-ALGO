package com.datastructures.Tree.Dfstraverse;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class DFSUsingBST
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

    public static void main(String[] args)
    {
        Scanner sc=new Scanner(System.in);
        int T=sc.nextInt();
        Node root=null;
        while(T-->0)
        {
            int data=sc.nextInt();
            root=insert(root,data);
        }
        levelOrder(root);
    }

    public static Node insert(Node root,int data)
    {
        if(root==null)
        {
            return new Node(data);
        }
        else
        {
            Node cur;
            if(data<=root.data)
            {
                cur=insert(root.left,data);
                root.left=cur;
            }
            else
            {
                cur=insert(root.right,data);
                root.right=cur;
            }
            return root;
        }
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
}
