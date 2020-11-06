package com.datastructures.Dequeue;

public class UseDeque
{
    public static void main(String[] args) {

        DoubleEndedQueue<Integer> deque = new DoubleEndedQueue<>();

        deque.addEnd(39);
        deque.addEnd(32);
        System.out.print(deque.removeBeg());
        System.out.print(deque.removeBeg());
        System.out.print(deque.removeBeg());

    }
}
