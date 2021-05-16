//Day 1 (1. Sort an array of 0’s 1’s 2’s without using extra space or sorting algo)

"https://www.youtube.com/watch?v=oaVa-9wmpns&list=PLgUwDviBIf0rPG3Ictpu74YWBQ1CaBkm2&index=3"

class Solution {
    public void sortColors(int[] nums) 
    {
        //dutch national flag
        int low=0,mid=0,end=nums.length-1;
        while(mid<=end)
        {
            if(nums[mid]==0)
            {
                int temp = nums[low];
                nums[low] = nums[mid];
                nums[mid] = temp;
                low++;
                mid++;
            }
            else if(nums[mid]==1)
            {
                mid++;
            }
            else if(nums[mid]==2)
            {
                int temp = nums[mid];
                nums[mid] = nums[end];
                nums[end] = temp;
                end--;
            }
        }
        for(int i=0; i<nums.length; i++)
        {
            System.out.print(nums[i]+" ");
        }
    }
}

//Day 1 (2. Repeat and Missing Number)

"https://www.youtube.com/watch?v=qfbBRtbhQ04&feature=youtu.be"

class Solve {
    int[] findTwoElement(int arr[], int size)
    {
        // code here
        
        int ar[] = new int[2];
        int i;
        for (i = 0; i < size; i++) 
        {
            int abs_val = Math.abs(arr[i]);
            if (arr[abs_val - 1] > 0)
                arr[abs_val - 1] = -arr[abs_val - 1];
            else
                ar[0]=abs_val;
        }
  
        for (i = 0; i < size; i++) 
        {
            if (arr[i] > 0)
                {
                    ar[1]=i+1;
                    break;
                }
        }
        return ar;
    }   
}

//OR



//Day 1 (3. Merge two sorted Arrays without extra space)


"https://www.youtube.com/watch?v=59VkIo4Pk3Y"
class Solution
{
    //Function to merge the arrays.
    public static void merge(long arr1[], long arr2[], int n, int m) 
    {
        int i = n-1, j=0;
        while (i >= 0 && j < m) 
        {
            if (arr1[i] > arr2[j])
            {
                long temp = arr2[j];
                arr2[j] = arr1[i];
                arr1[i] = temp;
                i--;
                j++;
            }
            else {
                break;
            }
        }
        Arrays.sort(arr1);
        Arrays.sort(arr2);
    }
}

//OR (Shell Sort)

"https://www.youtube.com/watch?v=hVl2b3bLzBw"


//Day 1 (4. Kadane's Algo)

class Solution 
{
    public int maxSubArray(int[] nums) 
    {    
        int sum=0,max=nums[0];
        for(int i=0; i<nums.length; i++)
        {
            sum+=nums[i];
            if(sum>max)
                max = sum;
            if(sum<0)
                sum=0;
        }
        return max;
    }
}

//Day 1 (5. Merge Overlapping Subintervals)

class Solution {
    public int[][] merge(int[][] intervals) {
        
        List<int[]> res = new ArrayList<>();
        if(intervals.length == 0 || intervals ==null)
            return res.toArray(new int[0][]);
        
        Arrays.sort(intervals, (a,b) -> a[0] - b[0]);
        
        int start = intervals[0][0];
        int end = intervals[0][1];
        
        for(int[] i :intervals)
        {
            if(i[0] <= end)
                end = Math.max(end,i[1]);
            else
            {
                res.add(new int[]{start, end});
                start = i[0];
                end = i[1];
            }
        }
        res.add(new int[]{start, end});
        return res.toArray(new int[0][]);
    }
}

//Day 1(6. Find the duplicate in an array of N+1 integers.)

//method 1 (sorting)
//method 2 (frequency array ---> if found 2 frequency then return)

//method 3 ( hare and tortoise) --->
class Solution {
  public int findDuplicate(int[] nums) 
  {
    int hare = nums[0];
    int tortoise = nums[0];
    do{
        hare = nums[hare];
        tortoise = nums[nums[tortoise]];
    }
    while(hare!=tortoise);
    tortoise = nums[0];
    while(hare!=tortoise)
    {
        hare = nums[hare];
        tortoise = nums[tortoise];
    }
    return hare;
  }
}

//Day 2 (1. Set Matrix Zeros )

//Method 1:-
//If matrix has only positive nos. then

class Solution {
    public void setZeroes(int[][] matrix) {

        for(int i=0; i<matrix.length; i++)
        {
            for(int j=0; j<matrix[0].length; j++)
            {
                if(matrix[i][j]==0)
                {
                    //for columns
                    for(int k=0; k<matrix[0].length; k++)
                    {
                        if(matrix[i][k]!=0)
                            matrix[i][k]=-1;
                    }
                    //for rows
                    for(int k=0; k<matrix.length; k++)
                    {
                        if(matrix[k][j]!=0)
                            matrix[k][j]=-1;
                    }
                }
            }
        }
        for(int i=0; i<matrix.length; i++)
        {
            for(int j=0; j<matrix[0].length; j++)
            {
                if(matrix[i][j]==-1)
                {
                   matrix[i][j]=0; 
                }
            }
        }
    }
}
//Method 2:-
//Using extra space

class Solution {
    public void setZeroes(int[][] matrix) {

        int[] columns = new int[matrix[0].length];
        int[] rows = new int[matrix.length];

        for(int i=0; i<matrix.length; i++)
        {
            for(int j=0; j<matrix[0].length; j++)
            {
                if(matrix[i][j]==0)
                {
                    columns[j]=-1;
                    rows[i]=-1;
                }
            }
        }
        for(int i=0; i<matrix.length; i++)
        {
            for(int j=0; j<matrix[0].length; j++)
            {
                if(rows[i]==-1 || columns[j]==-1)
                {
                   matrix[i][j]=0; 
                }
            }
        }
    }
}

//Method 3 without extra space

class Solution {
    public void setZeroes(int[][] matrix) {

        int col0=1, rows = matrix.length, cols = matrix[0].length;

        for (int i = 0; i < rows; i++) 
        {
            if (matrix[i][0] == 0) 
                col0 = 0;
            for (int j = 1; j < cols; j++)
                if (matrix[i][j] == 0)
                    matrix[i][0] = matrix[0][j] = 0;
        }

        for (int i = rows - 1; i >= 0; i--) 
        {
            for (int j = cols - 1; j >= 1; j--)
                if (matrix[i][0] == 0 || matrix[0][j] == 0)
                    matrix[i][j] = 0;
            if (col0 == 0) matrix[i][0] = 0;
        }
    }

//Day 2 (2. Pascal Triangle)

"https://youtu.be/1z4nW3_lSKI"

class Solution {
    public List<List<Integer>> generate(int numRows) {
        
        List<List<Integer>> res = new ArrayList<>();
        if(numRows == 0)
            return res;
        res.add(new ArrayList<>());
        res.get(0).add(1);
        
        for(int i=1; i<numRows; i++)
        {
            List<Integer> l = new ArrayList<>();
            l.add(1);
            for(int j=1; j<i; j++)
            {
                l.add(res.get(i-1).get(j-1) + res.get(i-1).get(j));
            }
            l.add(1);
            res.add(l);
        }
        return res;
        
    }
}

//Day 2 (3. Next Permutation)

// In this, we check for break point in the array 
// i.e 
// i=length-2;
// 1. a[i]<a[i+1] this is breakpoint, note this i index and 
// 2. again traverse from last to find element greater than a[i]. 
// 3. Once found geater element, swap them.
// 4. Now, reverse the array from i+1 to last index. So, this next permutation.

class Solution {
    public void nextPermutation(int[] nums) {
        
        int i = nums.length-2;
        while(i>=0 && nums[i] >= nums[i+1])
            i--;
        if(i>=0)
        {
            int j = nums.length-1;
            while(j>=0 && nums[j]<=nums[i])
                j--;
            swap(nums,i,j);
        }
        reverse(nums,i+1);
    }
    void swap(int[] nums,  int i ,int j)
    {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    void reverse(int[] nums, int i)
    {
        int j = nums.length-1;
        while(i<j)
        {
            swap(nums,i,j);
            i++;
            j--;
        }
    }
}

//Day 2 (4. Inversion of Array)

//https://youtu.be/sV4RhDIIKO0

static int mergeSortAndCount(long[] arr, int l, int r)
    {
        int count = 0;

        if (l < r) {
            int m = (l + r) / 2;
            //calculating inversion from left,right and merged array
            count += mergeSortAndCount(arr, l, m);
            count += mergeSortAndCount(arr, m + 1, r);
            count += mergeAndCount(arr, l, m, r);
        }

        return count;
    }
    static long mergeAndCount(long[] arr, int l,
                                     int m, int r)
    {
        //filling array left and and right with arr 

        long[] left = Arrays.copyOfRange(arr, l, m + 1);
        long[] right = Arrays.copyOfRange(arr, m + 1, r + 1);

        long i = 0, j = 0, k = l, swaps = 0;

        while (i < left.length && j < right.length) {
            if (left[(int)i] <= right[(int)j])
                arr[(int)k++] = left[(int)i++];
            else {
                arr[(int)k++] = right[(int)j++];
                swaps += left.length - i;  //this main condition to count inversion
            }
        }
        while (i < left.length)
            arr[(int)k++] = left[(int)i++];
        while (j < right.length)
            arr[(int)k++] = right[(int)j++];
        return swaps;
    }


//Day 2 (5. Stock Buy and Sell)

//like kadane's algo
//storing every minimum element in array and checking for maximum profit when arr[i]>min     
public int maxProfit(int[] prices) 
{
        
    //max = 0, because we need to return 0 in case of no profit    
    int max = 0, min = Integer.MAX_VALUE;        
    for(int i=0 ; i<prices.length; i++)
    {
        if(prices[i] < min)
           min = prices[i];  //storing minumum
       else //when prices[i]>min then it may be maximum to get more profit
       {
           max = Math.max(max, prices[i] - min); //check for maximum profit
       }
    }
    return max;
        
}    

//Day 3 (6. Rotate Matrix)

public void rotate(int[][] matrix) {
        
        //first transpose matrix
        for(int i=0 ;i<matrix.length; i++) //loop for row(size = 3 in tc)
        {
            //this loop is for column(size = 3 in tc)
            //j = i because everytime we are moving to next column
            for(int j=i; j<matrix[0].length; j++)
            {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        
        //now reverse elements of row
        for(int i=0 ;i<matrix.length; i++)
        {
            //taking matrix.length/2 for size of each row
            for(int j=0; j<matrix.length/2; j++)
            {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][matrix.length-1-j];
                matrix[i][matrix.length-1-j] = temp;
            }
        }
        
    }


//Day 8 (1. N meeting in one room)


//class for meeting where we have stored start,end and position of meeting
class Meeting
{
    int start;
    int end;
    int pos;
    
    Meeting(int start, int end, int pos)
    {
        this.start = start;
        this.end = end;
        this.pos = pos;
    }
}
class meetCompare implements Comparator<Meeting>
{
    @Override
    public int compare(Meeting o1, Meeting o2)
    {
        if(o1.end < o2.end)
        {
            return -1;
        }
        else if(o1.end > o2.end)
        {
            return 1;
        }
        //if end time of meetings are same then sort acc. to meeting's postion
        else if(o1.pos < o2.pos)
        {
            return -1;
        }
        return 1;
    }
}
class Solution 
{
    //Function to find the maximum number of meetings that can
    //be performed in a meeting room.
    public static int maxMeetings(int start[], int end[], int n)
    {
        ArrayList<Meeting> meet = new ArrayList<>();
        //filling our arraylist
        for(int i=0; i<start.length; i++)
        {
            //created new object of meeting to add in arraylist
            meet.add(new Meeting(start[i], end[i], i+1)); //i+1, as we want 1 based indexing 
        }
        
        //comparator class to sort meeting acc. to its end time
        meetCompare mc = new meetCompare();
        Collections.sort(meet,mc);
        
        //for result
        ArrayList<Integer> ans = new ArrayList<>();
        //added first meeting's postion(i.e 1) in ans
        ans.add(meet.get(0).pos);
        //noted ending time of first meeting
        int limit = meet.get(0).end;
        
        //starting from 1..as we have already added 1st meeting in ans
        //iterating this all meeting
        for(int i=1; i<meet.size(); i++)
        {
            //only if starting time of meeting > limit
            if(meet.get(i).start > limit)
            {
                limit = meet.get(i).end; //limit updated
                ans.add(meet.get(i).pos);
            }
        }
        return ans.size();
    }
}

