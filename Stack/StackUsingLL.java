package com.datastructures.Stack;

public class StackUsingLL<E> {
    static class Node<E>
    {
        E data;
        Node<E> next;
        public Node(E data)
        {
            this.data = data;
            next=null;
        }
    }
    Node<E> head;
    public void push(E data)
    {

        Node<E> newNode = new Node<>(data);
        Node<E> temp = head;
        if(head == null)
        {
            head = newNode;
        }
        else
        {
            while(temp.next!=null)
                temp=temp.next;
            temp.next=newNode;
        }
    }
    public E pop() throws Exception
    {
        Node<E> temp = head;
        if(temp==null)
        {
            throw new Exception("Stack is empty");
        }
        else if(temp.next==null)
        {
            Node<E> removeElem = head;
            head= null;
            return  removeElem.data;
        }
        else
        {
            while(temp.next.next!=null)
            {
                temp=temp.next;
            }
            Node<E> removeElem= temp.next;
            temp.next=null;
            return (E) removeElem.data;
        }
    }
    public E peek() throws Exception
    {
        Node<E> temp=head;
        if(temp==null)
        {
            throw new Exception("Stack is empty");
        }
        else
        {
            while(temp.next!=null)
                temp = temp.next;
            return temp.data;
        }
    }
}
