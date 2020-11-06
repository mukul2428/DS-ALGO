package com.datastructures.Dequeue;

public class DoubleEndedQueue<E> {

    public class Node
    {
        E data;
        Node next, prev;
        public Node(E data)
        {
            this.data = data;
            next = prev = null;
        }
    }
    Node head, tail;
    public void addEnd(E data)
    {
        Node newNode = new Node(data);
        if(head == null)
        {
            head = tail = newNode;
        }
        else
        {
            tail.next=newNode;
            newNode.prev=tail;
            tail=tail.next;
        }
    }
    public void addBeg(E data)
    {
        Node newNode = new Node(data);
        if(head==null)
        {
            head=tail=newNode;
        }
        else
        {
            head.prev=newNode;
            Node temp=head;
            head=head.prev;
            head.next=temp;
        }
    }
    public E removeBeg()
    {
        if(head==null)
        {
            return null;
        }
        else if(head.next==null)
        {
            Node temp = head;
            head = null;
            return temp.data;
        }
        else
        {
            Node temp = head;
            head=head.next;
            head.prev=null;
            return temp.data;
        }
    }
    public E removeEnd()
    {
        if(head==null)
        {
            return null;
        }
        else if(head.next==null)
        {
            Node temp = head;
            head = null;
            return temp.data;
        }
        else
        {
            Node temp = tail;
            tail=tail.prev;
            tail.next=null;
            return temp.data;
        }
    }
    public void traverse()
    {
        if(head==null)
        {
            System.out.println("Queue is empty");
        }
        else
        {
            Node temp = head;
            while(temp!=null)
            {
                System.out.print(temp.data+" ");
                temp=temp.next;
            }
        }
    }

}
