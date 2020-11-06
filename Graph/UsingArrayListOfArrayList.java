package com.datastructures.Graph;

import java.util.ArrayList;
import java.util.Scanner;

public class UsingArrayListOfArrayList
{

    public static void main(String[] args)
    {
        Scanner sc = new Scanner(System.in);

        System.out.println("Enter no. of vertices:");
        int ver = sc.nextInt();

        ArrayList<ArrayList<Integer>> arr = new ArrayList<>();

        for(int i=0; i<ver; i++)
        {
            arr.add(new ArrayList<>());
        }

        System.out.println("Enter no. of edges:");
        int e = sc.nextInt();

        for(int i=0; i<e; i++)
        {
            System.out.println("u->");
            int u = sc.nextInt();
            System.out.println("v->");
            int v = sc.nextInt();

            insert(arr,u,v);
        }
        print(arr);

    }

    private static void print(ArrayList<ArrayList<Integer>> arr)
    {
        for(int i=0; i<arr.size(); i++)
        {
            System.out.print(i);
            for(int j=0; j<arr.get(i).size(); j++)
            {
                System.out.print("->"+arr.get(i).get(j));
            }
            System.out.println();
        }
    }

    public static void insert(ArrayList<ArrayList<Integer>> arr, int u, int v)
    {
        arr.get(u).add(v);
        arr.get(v).add(u);
    }

}
