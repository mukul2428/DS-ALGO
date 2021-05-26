package com.datastructures.GraphTraversal;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class BFSTraversal {

    public static void main(String[] args)
    {
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

        //bfs

        ArrayList<Integer> print = new ArrayList<>();
        //V+1 for 1 based indexing nodes---> nodes start from 1 to V not 0 to V-1
        boolean[] bool = new boolean[node + 1];
        //using loop inorder to check every node is traversed using bfs or not
        //helpful for unconnected graph
        for(int i = 1; i <= node; i++)
        {
            //check if current node is visited or not
            if(!bool[i])
            {
                Queue<Integer> q = new LinkedList<>();

                q.add(i);
                bool[i] = true;

                while(!q.isEmpty())
                {
                    //take out node from queue and mark it as visited
                    int nd = q.poll();
                    print.add(nd);

                    //traverse neighbour nodes of curr nodes
                    for(Integer neighbour : ans.get(node))
                    {
                        if(!bool[neighbour])
                        {
                            q.add(neighbour);
                            bool[neighbour] = true;
                        }
                    }
                }
            }
        }
        System.out.println(print);
    }
}
