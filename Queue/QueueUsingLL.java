package com.datastructures.Queue;

public class QueueUsingLL<E> {

    public class Node
    {
        E data;
        Node next;
        public Node(E data)
        {
            this.data = data;
            next = null;
        }
    }
    Node head,rear;
    public void enqueue(E data)
    {
        Node newNode = new Node(data);
        if (head == null)
        {
            head = rear = newNode;
        }
        else
        {
            rear.next=newNode;
            rear = rear.next;
        }
    }
    public E dequeue() throws Exception
    {
        if(head==rear)
        {
            throw new Exception("Queue is empty");
        }
        else if(head.next==null)
        {
            Node delete = head;
            head=null;
            return delete.data;
        }
        else
        {
            Node delete = head;
            head = head.next;
            return delete.data;
        }
    }
    public void traverse()
    {
        Node temp = head;
        if(head == null)
            System.out.println("Queue is empty");
        else
        {
            while (temp!=null)
            {
                System.out.print(temp.data+" ");
                temp= temp.next;
            }
        }
    }

}
