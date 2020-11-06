package com.datastructures.Queue;

public class UsingQueue {

    public static void main(String[] args) throws Exception {
        QueueUsingLL<Integer> queue = new QueueUsingLL<>();
        for(int i=1; i<=10; i++)
            queue.enqueue(i);
        for(int i=1; i<=3;i++)
            System.out.print(queue.dequeue());
        queue.traverse();
    }

}
