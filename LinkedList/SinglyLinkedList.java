package com.datastructures.LinkedList;

public class SinglyLinkedList<E>
{
    static class Node<E>
    {
        E data;
        Node next;
        public Node(E data)
        {
            this.data = data;
            next=null;
        }
    }
    Node<E>head;
    public void insertTop(E data)
    {
        Node<E> newNode = new Node<>(data);
        if(head == null)
        {
            head = newNode;
        }
        else
        {
            newNode.next=head;
            head=newNode;
        }
    }
    public void insertEnd(E data)
    {
        Node<E> newNode = new Node<E>(data);
        if(head == null)
        {
            head = newNode;
        }
        else
        {
            Node<E> temp = head;
            while(temp.next!=null)
                temp=temp.next;
            temp.next=newNode;
        }
    }
    public void InsertSpecific(E data, int pos) throws Exception
    {
        Node<E> newNode = new Node<E>(data);
        if(pos<=0)
        {
            throw new Exception("Position cannot be less than one");
        }
        else if(pos == 1)
        {
            if(head==null)
            {
                head=newNode;
            }
            else
            {
                Node<E> temp = head;
                head = newNode;
                head.next=temp;
            }
        }
        else
        {
            if(head==null)
            {
                head=newNode;
            }
            else
            {
                Node<E> temp = head;
                Node<E> ptr = head;
                for(int i=1; i<pos && temp!=null; i++)
                {
                    ptr = temp;
                    temp=temp.next;
                }
                if(temp==null)
                {
                    ptr.next=newNode;
                }
                else
                {
                    ptr.next=newNode;
                    newNode.next=temp;
                }
            }
        }

    }
    public void traverse()
    {
        Node<E> temp = head;
        while(temp!=null)
        {
            System.out.print(temp.data+" ");
            temp=temp.next;
        }
    }
    public void deleteTop() throws Exception
    {
        Node<E> temp = head;
        if(head==null)
        {
            throw new Exception("Cannot delete");
        }
        else if(head.next==null)
        {
            head=null;
        }
        else
        {
            head=head.next;
        }
    }
    public void deleteSpecific(int pos) throws Exception
    {
        if(pos<=0)
            throw new Exception("Position should be greater than zero");
        else if(pos==1)
        {
            if(head==null)
            {
                throw new Exception("Deletion is not possible");
            }
            else if(head.next==null)
            {
                head=null;
            }
            else
            {
                head=head.next;
            }
        }
        else
        {
            if(head==null)
            {
                throw new Exception("Deletion is not possible");
            }
            else if(head.next==null)
            {
                head=null;
            }
            else
            {
                Node<E> temp = head, ptr = head;
                for(int i=1; i<pos && temp.next!=null; i++)
                {
                    ptr=temp;
                    temp=temp.next;
                }
                if(temp.next==null)
                {
                    ptr.next=null;
                }
                else
                {
                    ptr.next=temp.next;
                }
            }
        }

    }
    public void deleteEnd() throws Exception
    {
        Node<E> temp = head;
        if(head==null)
        {
            throw new Exception("Cannot delete");
        }
        else if(head.next==null)
        {
            head=null;
        }
        else
        {
            while(temp.next.next!=null)
            {
                temp=temp.next;
            }
            temp.next=null;
        }
    }
}
