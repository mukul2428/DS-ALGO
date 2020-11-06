package com.datastructures.Stack;

public class UseStack {
    public static void main(String[] args) throws Exception {
        StackUsingLL<Integer> stack = new StackUsingLL<>();
        for(int i=0; i<=10; i++)
        stack.push(i);
        System.out.println(stack.pop());
        System.out.println(stack.peek());
        System.out.println(stack.peek());
    }
}
