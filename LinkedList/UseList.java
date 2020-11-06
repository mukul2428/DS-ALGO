package com.datastructures.LinkedList;

public class UseList {

    public static void main(String[] args) throws Exception {

        SinglyLinkedList<Integer> myList = new SinglyLinkedList<>();

        for(int i=1 ;i<=10; i++)
            myList.insertEnd(i);
        myList.deleteTop();
        myList.deleteTop();
        myList.traverse();

    }
}
