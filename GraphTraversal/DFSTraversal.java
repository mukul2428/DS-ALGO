package com.datastructures.GraphTraversal;

import java.util.ArrayList;
import java.util.Scanner;
import java.util.Stack;

public class DFSTraversal {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);
        int node = sc.nextInt(); //no.of nodes

        ArrayList<ArrayList<Integer>> ans = new ArrayList<>();
        //for 1 base indexing i.e starting vertex from 1 to V
        for(int i = 0; i <= node; i++)
        {
            ans.add(new ArrayList<>());
        }
        int edges = sc.nextInt(); //no. of edges
        //making graphs by joining vertices with edges
        for(int i = 1; i <= edges; i++)
        {
            System.out.println("Enter u and v");
            int u = sc.nextInt();
            int v = sc.nextInt();
            ans.get(u).add(v);
            ans.get(v).add(u);
        }

        //dfs
        ArrayList<Integer> print = new ArrayList<>();
        //n + 1 for 1 based indexing i.e node start from 1 to V
        boolean[] bool = new boolean[node + 1];

        //loop helpful for unconnected graph
        //for 1 based indexing i.e node start from 1 to V
        for(int i = 1; i <= node; i++)
        {
            if(!bool[i])
            {
                //recur(print, bool, ans, i);
                usingStack(print, bool, ans, i);
            }
        }
        System.out.println(print);

    }

    private static void usingStack(ArrayList<Integer> print, boolean[] bool, ArrayList<ArrayList<Integer>> ans, int i) {

        Stack<Integer> stack = new Stack<>();
        stack.add(i);
        bool[i] = true;

        while(!stack.isEmpty())
        {
            int nd = stack.pop();
            print.add(nd);
            //for neighbours
            for(Integer neighbour : ans.get(nd))
            {
                if(!bool[neighbour])
                {
                    stack.add(neighbour);
                    bool[neighbour] = true;
                }
            }
        }
    }

    private static void recur(ArrayList<Integer> print, boolean[] bool, ArrayList<ArrayList<Integer>> ans, int i)
    {

        //check for adjacent nodes
        for(Integer nd : ans.get(i))
        {
            if(!bool[nd])
            {
                bool[nd] = true;
                print.add(nd);
                recur(print, bool, ans, nd);
            }
        }
    }
}
