package com.datastructures.LinkedList;

//public class CircularLL
//{
//    static class Node<E>
//    {
//        E data;
//        Node next;
//        public Node(E data)
//        {
//            this.data=data;
//            next=null;
//        }
//    }
//    Node head;
//    public void insertFirst(E data)
//    {
//        Node newNode = new Node<>(data);
//        if(head==null)
//        {
//            head=newNode;
//            head.next=head;
//        }
//        else
//        {
//            newNode.next=head;
//        }
//    }
//    public void insertLast(E data)
//    {
//        Node newNode = new Node<>(data);
//        Node temp=head;
//        if(head==null)
//        {
//            head=newNode;
//            head.next=head;
//        }
//        else
//        {
//            while(temp.next!=head)
//            {
//                temp=temp.next;
//            }
//            temp.next=newNode;
//        }
//    }
//}
