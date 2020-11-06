package com.datastructures.LinkedList;

import java.util.Scanner;

public class RemoveDuplicate
{
    static class Node
    {
        int data;
        Node next;
        Node(int d)
        {
            data=d;
            next=null;
        }
    }

    public static void main(String[] args)
    {
        Scanner sc=new Scanner(System.in);
        Node head = null;
        int T = sc.nextInt();
        while(T-->0)
        {
            int ele=sc.nextInt();
            head=insert(head,ele);
        }
        head=removeDuplicates(head);
        display(head);

    }

    public static Node insert(Node head, int data)
    {
        Node newNode = new Node(data);
        if (head == null)
            head = newNode;
        else if (head.next == null)
            head.next = newNode;
        else
        {
            Node start = head;
            while (start.next != null)
                start = start.next;
            start.next = newNode;
        }
        return head;
    }

    //only for sorted linked list
    public static Node removeDuplicates(Node head)
    {
        if (head == null)
            return null;
        Node s = head;
        while (s.next != null)
        {
            if (s.data == s.next.data)
                s.next = s.next.next;
            else
                s = s.next;
        }
        return head;
    }

    public static void display(Node head)
    {
        Node start = head;
        while (start != null) {
            System.out.print(start.data + " ");
            start = start.next;
        }
    }
}
