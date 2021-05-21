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

//Day 2 (6. Rotate Matrix)

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

//Day 3 (1.Search in 2D matrix)
//Leetcode

public boolean searchMatrix(int[][] matrix, int target) {
        
        //used binary search without extra space
        int n = matrix.length;
        int m = matrix[0].length;
        
        int beg = 0;
        int end = (n * m) - 1;
        
        //arr[][] = {1,3,4},{9,6,3},{1,5,3}
        //           0 1 2   3 4 5   6 7 8
        //beg = 0, end = 8, mid = 4
        // mid / m --> 4/3 = 1 and 4%3 = 1. So index will be (1,1) for mid = 4
        
        while(beg <= end)
        {
            int mid = beg + (end-beg)/2;
            
            //mid/m to find row and mid%m for column
            if(matrix[mid / m][mid % m] == target)
            {
                return true;
            }
            else if(matrix[mid / m][mid % m] > target)
            {
                end = mid - 1;
            }
            else
            {
                beg = mid + 1;
            }
        }
        return false;
        
    }

//gfg

public static int matSearch(int mat[][], int N, int M, int X)
    {   
        
        //we have started from s to traverse the matrix because every element below it is large and left of it is small  
        //               s 
        //mat[][] = {1,3,4}
        //          {5,6,7}
        //          {8,9,9}      
        
        int j = M - 1, i = 0;
        while(i < N && j >= 0)
        {
            if(mat[i][j] > X)
            {
                j--;
            }
            else if(mat[i][j] < X)
            {
                i++;
            }
            else
            {
                return 1;
            }
        }
        return 0; //not found when matrix will be out of bound
    }        


//Day 3 (2. Pow(X,n))

public double myPow(double x, int n) {
        
        double ans = 1.0;
        
        long nn = n; //taking long because while making negative a positive integer in case of Integer.MIN_VALUE the range of int will overflow
        
        //for negative case, make power positive
        if(nn < 0)
        {
            nn = -nn;
        }
        while(nn > 0)
        {
            //if power is even --> 2^4 = (2*2)^2
            if(nn % 2 == 0)
            {
                x = x*x;
                nn = nn/2;
            }
            //if power is odd --> 2^5 = 2*(2)^4
            else
            {
                ans = ans * x;
                nn = nn - 1;
            }
        }
        if(n < 0)
        {
            // for 2^-10 --> 1/2^10
            ans = (double) 1.0 / (double) ans;
        }
        return ans;
    }
    
//Day 3 (3. Majority Element (>N/2 times)) 

Method 1. O(n2) - Count frequency of elements and larger freq. element is our ans
Method 2. O(n) and O(n) space - frequency using hashmap

Method 3. Moore Voting Algo O(N) & O(1) Space
//explained in notes

 public int majorityElement(int[] nums) {

        int count = 0, element = 0; 

        for(int i = 0; i < nums.length; i++)
        {
            if(count == 0)
            {
                element = nums[i];
                count++;
            }
            else if(element != nums[i])
                count--;
            else
                count++;
        }
        //majority element
        return element;
    }  


//Day 3 (4. Majority Element (>N/3 times))


//In linear time and in O(1) space
//Using Boyer Moore Algo

public List<Integer> majorityElement(int[] nums) {
        
        //in this we can have 2 majority elements. so, we will take two variables
        
        int num1 = 0, num2 = 0, ct1 = 0, ct2 = 0;
        
        for(int i = 0; i < nums.length; i++)
        {
            //take care of series of if else statement

            //current element may be equal to num1
            //ith element may be equal to num1
            if(num1 == nums[i])
            {
                ct1++;
            }
             //for second largest element
            else if(num2 == nums[i])
            {
                ct2++;
            }
            else if(ct1 == 0)
            {
                num1 = nums[i];
                ct1++;
            }
            else if(ct2 == 0)
            {
                num2 = nums[i];
                ct2++;
            }
            //we got third element which is neither equal to num1 nor num2
            else
            {
                ct1--;
                ct2--;
            }
        }
        
        //now we have to check whether our first and second largest elements are greater than floor of n/2
        
        List<Integer> li = new ArrayList<>();
        
        int count1 = 0, count2 = 0;
        for(int i = 0; i<nums.length; i++)
        {
            if(nums[i] == num1)
            {
                count1++;
            }
            else if(nums[i] == num2)
            {
                count2++;
            }
        }
        
        if(count1 > (int)Math.floor(nums.length/3))
            li.add(num1);
        if(count2 > (int)Math.floor(nums.length/3))
            li.add(num2);
        
        return li; 
    }


//Day 3(5. Grid Unique Paths)  


Method 1. Recursion-->TLE

public int uniquePaths(int m, int n) {

        //using recursion, checking all possible paths
        int i = 0, j = 0;
        return recur(i, j, m, n);
    }
    int recur(int i, int j, int m, int n)
    {
        //when robot reached finish point
        if(i == m - 1 && j == n - 1)
        {
            return 1; //return 1 whenever we reached the finish
        }
        //condition when pointer moves out of bound without reaching finish point
        if(i >= m || j >= n)
        {
            return 0;
        }
        //sum of both right and left part of state space tree to get no. of unique paths
               //moving bottom     //moving right
        return recur(i+1, j, m, n) + recur(i, j+1, m, n);
    }

Method 2. DP --> O(m*n) space and time both

public int uniquePaths(int m, int n) {

        //using dp, checking all possible paths and storing same in table
        int i = 0, j = 0;
        int[][] dp = new int[m][n];
        for(int p = 0; p<m; p++)
        {
            for(int q = 0; q<n; q++)
            {
                dp[p][q] = -1;
            }
        }
        return recur(i, j, m, n, dp);
    }
    int recur(int i, int j, int m, int n, int[][] dp)
    {
        //when robot reached finish point
        if(i == m - 1 && j == n - 1)
        {
            return 1; //return 1 whenever we reached the finish
        }
        //condition when pointer moves out of bound without reaching finish point
        if(i >= m || j >= n)
        {
            return 0;
        }
        //if sum is same i.e we have already processed it
        if(dp[i][j] != -1)
            return dp[i][j];
        //sum of both right and left part of state space tree to get no. of unique paths
                              //moving bottom     //moving right
        else return dp[i][j] = recur(i+1, j, m, n, dp) + recur(i, j+1, m, n, dp);
    }

Method 3. Using combination O(m-1)

public int uniquePaths(int m, int n) {

        //used nCr formula
        int r = n-1, N = n+m-2; //total steps robot can take
        //either (n+m-2)C(n-1) or (n+m-2)C(m-1)
        long ans = 1;
        for(int i = 1; i <= r; i++)
        {
            ans = ans * (long) (N - r + i) / i;
        }
        return (int) ans;
    }  


//Day 3(6. Reverse Pairs)
//similar as inversion count
public int reversePairs(int[] nums) {
        
       //during merge sort check for pairs 
       return mergeSort(nums, 0, nums.length - 1);
        
    }
    
    int mergeSort(int nums[], int beg, int end)
    {
        int ans = 0;
        if(beg < end)
        {
            int mid = (beg + end) / 2;
            ans = mergeSort(nums, beg, mid);
            ans += mergeSort(nums, mid + 1, end);
            ans += merge(nums, beg, mid, end);
        }
        return ans;
    }
    int merge(int nums[], int beg, int mid, int end)
    {
        // first calculate ans then do further merging 
        //we keep on iterating right array till our condition is satisfied
        //stops when nums[i] <= 2 * num[j]
        
        int ans = 0, j = mid + 1;
        for(int i = beg; i <= mid; i++) //for next iteration of i, j will start from its previous position only. 
        {
            while(j <= end && nums[i] > (2 * (long) nums[j]))
                j++;
            //if we reached out of bound or our condition failed
            ans += (j - (mid+1));
        }
        
        //now do merging step
        
        
        ArrayList<Integer> arr = new ArrayList<>();
        int left = beg, right = mid + 1;
        while(left <= mid && right <= end)
        {
            if(nums[left] <= nums[right])
            {
                arr.add(nums[left++]);
            }
            else
            {
                arr.add(nums[right++]);
            }
        }
        while(left <= mid)
        {
            arr.add(nums[left++]);
        }
        while(right <= end)
        {
            arr.add(nums[right++]);
        }
        for(int p = beg; p <= end; p++)
        {
            nums[p] = arr.get(p - beg);
        }
        
//                          //+1 ,0 based indexing 
//         int m = mid - beg + 1;
//         int n = end - mid;
        
//         int[] left = new int[m];
//         int[] right = new int[n];
        
//         for(int i = 0; i < m; i++)
//         {
//             left[i] = nums[beg + i];
//         }
//         for(int i = 0; i < n; i++)
//         {
//             right[i] = nums[mid + 1 + i];
//         }
           
//         int k = beg, i = 0, j1 = 0;
//         while (i < m && j < n) 
//         {
//             if (left[i] <= right[j]) {
//                 nums[k] = left[i];
//                 i++;
//             }
//             else {
//                 nums[k] = right[j1];
//                 j1++;
//             }
//             k++;
//         }
        
//         while (i < m) {
//             nums[k] = left[i];
//             i++;
//             k++;
//         }
 
//         while (j1 < n) {
//             nums[k] = right[j1];
//             j1++;
//             k++;
//         }
        return ans;    
    
        
 
//Day 4 (1. Two Sum)


Method 1: using O(n2)

Method 2: using hashMap

public int[] twoSum(int[] nums, int target) {

        
    }Map<Integer, Integer> map = new HashMap<>();
        int[] ans = new int[2];
        for(int i = 0; i < nums.length; i++)
        {
            if(!map.containsKey(nums[i]))
            {
                map.put(target - nums[i], i);
            }
            else 
            {
                ans[0] = i;
                ans[1] = map.get(nums[i]);
                break;
            }
        }
        return ans;


//Day 4 (2. Four Sum)        


Method 1. Using 3 pointers--.> O(n3logn + nlogn)

1. Sort the array
2. 4th element = target - sum of 3. So, search this fourth element using binary search.
3. i = 0, j = i+1, k = j+1 and start binary search from k+1
4. Use HashSet to avoid dupliactes

Method 2. O(n3) Using two pointers --> O(n3)

public List<List<Integer>> fourSum(int[] nums, int target) {
        
        List<List<Integer>> ans = new ArrayList<>();
        
        Arrays.sort(nums);
        
        //taking pointers two find sum of first elements
        for(int i = 0; i < nums.length; i++)
        {
            for(int j = i + 1; j < nums.length; j++)
            {
                int t = target - (nums[i] + nums[j]);
                //taking two more pointers to check for t sum
                int left = j + 1, right = nums.length - 1;
                while(left < right)
                {
                    if(t < nums[left] + nums[right])
                        right--;
                    else if(t > nums[left] + nums[right])
                        left++;
                    //found all four elements
                    else
                    {
                        List<Integer> arr = new ArrayList<>();
                        arr.add(nums[i]);
                        arr.add(nums[j]);
                        arr.add(nums[left]);
                        arr.add(nums[right]);
                        ans.add(arr);
                        
                        //if 3rd and 4th element is added to list to give sum 
                        //as target then avoid duplicates of these elements 
                        //for left
                        while(left < right && nums[left] == arr.get(2))
                        {
                            left++;
                        }
                        //for right
                        while(left < right && nums[right] == arr.get(3))
                        {
                            right--;
                        }
                    }
                }
                //avoid duplicates for i and j also
                //took j+1 bcz when loop while update j value will increase
                while(j + 1 < nums.length && nums[j] == nums[j+1])
                {
                    j++;
                }
            }
            while(i + 1 < nums.length && nums[i] == nums[i+1])
            {
                i++;
            }
        }
        return ans;


//Day 4 (3. Longest Consecutive sequence)


O(n3) --> TLE

public int longestConsecutive(int[] nums) 
    {
        int longestStreak = 0;

        for (int num : nums) 
        {
            int currentNum = num;
            int currentStreak = 1;

            while (arrayContains(nums, currentNum + 1)) 
            {
                currentNum += 1;
                currentStreak += 1;
            }

            longestStreak = Math.max(longestStreak, currentStreak);
        }
         return longestStreak;
    }

 private boolean arrayContains(int[] arr, int num) 
     {
        for (int i = 0; i < arr.length; i++)
        {
            if (arr[i] == num) {
                return true;
            }
        }
        return false;
    }
Method 2 :- Using Set ---> O(3N)

public int longestConsecutive(int[] nums) 
    {
        //to avoid duplicate used set
        Set<Integer> set = new HashSet<>();
        int max = 0;

        //add all values to set
        for(int i = 0; i < nums.length; i++)
        {
            set.add(nums[i]);
        }

        for(int i = 0; i < nums.length; i++)
        {
            //we want minimum element of array which makes longest subsequence
            //so, whenever we get any element whose lesser value doesn't exist
            //then we start iteration from it
            if(!set.contains(nums[i] - 1))
            {
                int currValue = nums[i];
                int count = 1;

                //checking if greater value of current exists or not
                //checking all greater values
                while(set.contains(currValue + 1))
                {
                    currValue++;
                    count++;
                }
                max = Integer.max(max, count);
            }
        }
        return max;
    } 
    
    

//Day 4 (4. Largest Subarray with 0 sum)

Method 1. O(n2)
int maxLen(int arr[], int n)
    {
        // Your code here
        int len = 0;
        for(int  i = 0; i < n; i++)
        {
            int sum = 0;
            for(int j = i; j < n; j++)
            {
                sum += arr[j];
                if(sum == 0)
                {
                    len = Math.max(len, j - i + 1);
                }
            }
        }
        return len;
    }


Method 2. Using HashMap --> O(nlogn) and O(n) space
//explained in notes
    int maxLen(int arr[], int n)
    {
        
        //map is used to store the sum of subarrays and index till we get that subarray
        Map<Integer, Integer> map = new HashMap<>();
        
        int sum = 0, max = 0;
        for(int i = 0; i < n; i++)
        {
            sum += arr[i];
            
            //if sum is zero for simple iteration
            if(sum == 0)
            {
                max = i + 1; 
            }
            else
            {
                //if map already contains current sum
                if(map.containsKey(sum))
                {
                    //(current index - previous index in map) will give us length of subarray having sum = 0        
                    max = Math.max(max, i - map.get(sum));
                }
                //if map doesn't contains that sum
                else
                {
                    map.put(sum, i);
                }
            }
        }
        return max;
    }             


//Day 4 Hard (5. Count number of subarrays with given XOR)


 public int solve(ArrayList<Integer> A, int B) 
 {
        int count=0;
        int xorr=0;
        HashMap<Integer,Integer> freq=new HashMap<Integer, Integer>();
        
        //Y = xorr ^ B
        for(int i=0;i<A.size();i++)
        {
            xorr^=A.get(i);
            //Y = xorr ^ B
            
            //as number of times B occurred is equal to number of times Y (i.e xorr ^ B) occurred
            //so, this condition is to check whether there is any subarray of k xor present in between our xorr
            if(freq.get(xorr^B)!=null)
            {
                count+=freq.get(xorr^B);
            }
            //directly we found B then increase count
            if(xorr==B)
            {
                count++;
            }
            //add all xor of subarrays and their frequencies in map
            if(freq.get(xorr)!=null)
            {
                freq.put(xorr,freq.get(xorr)+1);
            }
            else{
                freq.put(xorr,1);
            }
        }
        return count;
    }  


//Day 4 (6. Longest substring without repeat)

Method 1. Using HashMap 
HashMap<Character, Integer> map = new HashMap<>();
         
        //took two pointers 
        int left = 0, right = 0, max = 0;
        //fixed left pointer and moving right
        while(right < S.length())
        {
            //if repeat element is found in map then move left pointer
            if(map.containsKey(S.charAt(right)))
            {
                //check if element found in map but that doesn't include in our current substring
                left = Math.max(map.get(S.charAt(right)) + 1, left);
            }
            
            //put every char in map
            map.put(S.charAt(right), right);
            //calculate max length
            max = Math.max(max, right - left + 1);
            
            right++;
        }
        return max;

Method 2: Using Set

public int lengthOfLongestSubstring(String s) 
    {
        int n = s.length();
        Set<Character> set = new HashSet<>();
        int ans = 0, i = 0, j = 0;
        while (i < n && j < n) 
        {
            if (!set.contains(s.charAt(j))){
                set.add(s.charAt(j++));
                ans = Math.max(ans, j - i);
            }
            else {
                set.remove(s.charAt(i++));
            }
        }
        return ans;
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

//Day 8 (2. Minimum number of platform required for railway)

//Function to find the minimum number of platforms required at the
    //railway station such that no train waits.
    static int findPlatform(int arrival[], int dept[], int n)
    {
        // add your code here
        Arrays.sort(arrival);
        Arrays.sort(dept);
        
        int platform_needed = 1; //initially first train will need one platform
        int min_platform = 1;
        
        int i = 1, j = 0; //i=1,as we will check arrival time of second train
                          //j=0, we will check departure time of first train
        
        while(i<arrival.length && j<dept.length)
        {
            //departure time is more
            if(arrival[i] <= dept[j]) //"=" because if arrival and dept are same then we need one more plaform
            {
                //a platform is already filled. so we need another plaform
                platform_needed++;
                i++; //train arrived, now move to another one
            }
            else //arrival time is more. so need to depart last train or vacant last platform
            {
                platform_needed--;
                j++;
            }
            //plaform required
            min_platform = Math.max(min_platform, platform_needed);
        }
        return min_platform;
    }

//Day 8 (3. Job sequencing problem)

//Function to find the maximum profit and the number of jobs done.
    int[] JobScheduling(Job arr[], int n) //this is class job array, job class has id, profit and deadline
    {
        //sort array in decreasing order using comparator 
        Arrays.sort(arr, (a,b) -> (b.profit - a.profit));
        
        //find max. deadline
        int max = 0;
        for(int i=0; i<n; i++)
        {
            if(arr[i].deadline > max)
            max = arr[i].deadline;
        }
        //array of size maximum deadline
        int[] x = new int[max+1]; 
        
        //i=1 as job id start from 1
        for(int i = 1; i<=max; i++)
        {
            x[i] = -1;
        }
        
        int jobProfit = 0, countJobs = 0;
        
        //iterating over Job array(checking every job)
        for(int i=0; i<n; i++)
        {
            //starting from deadline of every job till 0
            for(int j=arr[i].deadline; j>0; j--)
            {
                //if we get any vacant space then put job there
                if(x[j] == -1)
                {
                    x[j] = arr[i].id;
                    jobProfit += arr[i].profit; //increased total profit by current job's profit
                    countJobs++; //increased count of job
                    break;
                }
            }
        }
        int ans[] = new int[2];
        ans[0] = countJobs;
        ans[1] = jobProfit;
        
        return ans;
    }   

//Day 8 (4. Fractional Knapsack Problem)

class Item {
    int value, weight;
    Item(int x, int y){
        this.value = x;
        this.weight = y;
    }
}

//comparator to sort value/weight in decreasing order
class sortComparator implements Comparator<Item>
{
    @Override
    public int compare(Item a, Item b)
    {
        double x = (double)(a.value) / (double)(a.weight);
        double y = (double)(b.value) / (double)(b.weight);
        if(x < y)
        {
            return 1;
        }
        else if(x > y)
        {
            return -1;
        }
        else return 0;
    }
}

class Solution
{
    //Function to get the maximum total value in the knapsack.
    double fractionalKnapsack(int W, Item arr[], int n) 
    {
        // Your code here
        
        Arrays.sort(arr, new sortComparator());
        int currWeight = 0; 
        double finalValue = 0.0;
        
        for(int i = 0; i < n ; i++)
        {
            if(currWeight + arr[i].weight <= W) //if ith item + currWeight is less than knapsack weight
            {
                currWeight += arr[i].weight; //increase currWeight
                finalValue += arr[i].value;
            }
            else //taking fraction of value and break;
            {
                int rem = W - currWeight; //check remaining capacity of knapsack
                finalValue += ((double)arr[i].value / (double)arr[i].weight) * (double)rem;
                break;
            }
            
        }
        return finalValue;
    }
} 

//Day 8 (5. Greedy algorithm to find minimum number of coins)

static void findMin(int V)
{
    ArrayList<Integer> arr = new ArrayList<>();

    int[] r = {1,2,5,10,20,50,100,500,1000};

    for(int i = n-1; i>=0; i--)
    {
        while(V >= r[i])
        {
            V -= r[i];
            arr.add(r[i]);
        }
    }
    System.out.println(arr);
}     
    
//Day 9 (1. Subset sums)

ArrayList<Integer> subsetSums(ArrayList<Integer> arr, int N){
        
        ArrayList<Integer> ans = new ArrayList<>();
        
        //passed index of array and sum initially as 0
        recur(0,0,ans,arr,N);
        
        //we need sorted ans
        Collections.sort(ans);
        
        return ans;
    }
    
    void recur(int index, int sum, ArrayList<Integer> ans, ArrayList<Integer> arr, int N)
    {
        //if index goes out of bound of array then add the sum to ans
        if(index == N)
        {
            ans.add(sum);
            return;
        }
        //picked up the element and increased the sum by element on that index
        recur(index+1, sum + arr.get(index), ans, arr, N);
        //didn't picked the element
        recur(index+1, sum, ans, arr, N);
        
    } 

//Day 9 (2. Subset 2)

public List<List<Integer>> subsetsWithDup(int[] nums) {
        
        //sorted so that we can check for previous duplicate element in array
        Arrays.sort(nums);
        List<List<Integer>>ans = new ArrayList<>();
        // passed index = 0 and innerList
        recur(0, new ArrayList<>(), ans, nums);
        
        return ans;
    }
    
    void recur(int index, List<Integer> innerList, List<List<Integer>> ans, int[] nums)
    {
        //add innerlist to ans list, initially innerlist will be empty([])
        ans.add(new ArrayList<>(innerList));
        
        //this loop will run from specific index till end
        // {3,2,4,1} --> from index 1 to end we can have many subsets like {2,4}, {2,1} etc
        
        for(int i = index; i < nums.length; i++)
        {
            //don't add duplicate elements in the list
            if(i != index && nums[i] == nums[i-1])
            {
                continue;
            }
            //add elements to innerlist if not duplicate
            innerList.add(nums[i]);
            
            recur(i+1, innerList, ans, nums);
            
            innerList.remove(innerList.size() - 1);
            
        }
    }   
    
 
    
//Day 9 (3. Combination sum 1)


Using Recursion --> O(2^t * k), k is number of innerlist

  public List<List<Integer>> combinationSum(int[] candidates, int target) {

        List<List<Integer>> ans = new ArrayList<>();

        //passed index for array(as 0) and innerlist
        return recur(0, target, candidates, ans, new ArrayList<>());

    }

    List<List<Integer>> recur(int index, int target, int[] arr, List<List<Integer>> ans, List<Integer> ds)
    {
        //this base case i.e we got our combination
        if(index == arr.length)
        {
            //we got combination now add to ans list
            if(target == 0)
            {
                //this will be linear time
                ans.add(new ArrayList<>(ds));
            }
            return ans;
        }
        //searching for our combination
        //case when we are taking current element
        if(arr[index] <= target)
        {
            //add curr element to list
            ds.add(arr[index]);
            //i will remain same, as we can take same number any no. of times
            recur(index, target - arr[index], arr, ans, ds);
            //after recursion is over then remove last elements from innerlist
            ds.remove(ds.size()-1);
        }
        //case when we are not considering curr elements, now increase index value
        recur(index + 1, target, arr, ans, ds);
        return ans;
    }  
    

  
//Day 9 (4. Combination sum 2)

public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        
        List<List<Integer>> ans = new ArrayList<>();
        
        //first sort the array bcz we need to remove duplicate elements
        Arrays.sort(candidates);
        
        //index is passed as 0 and new innerList is passed
        //we passes the index because whenever we go deep in recursion we need
        //the index from which we need to pick the element
        recur(0, target, candidates, ans, new ArrayList<>());
        
        return ans;
    }
    
    void recur(int index, int t, int[] arr, List<List<Integer>> ans, List<Integer> ds)
    {
        //base case 
        if(t == 0)
        {
            ans.add(new ArrayList<>(ds));
            return;
        }
        //everytime starting from specific index till last
        for(int i = index; i < arr.length; i++)
        {
            //i > index means that we need not to pick duplicate element
            if(i > index && arr[i] == arr[i-1])
            {
                continue;
            }
            //break the recursion if current element is greater than target
            if(arr[i] > t)
            {
                break;
            }
            //add curr element to list to make combination
            ds.add(arr[i]);
            
            //recursive call for i+1
            recur(i + 1, t - arr[i], arr, ans, ds);
            
            //remove last element of innerList after recursive call
            ds.remove(ds.size() - 1);
        }
    }

//Day 9 (5. Palindrome Partitioning)



public List<List<String>> partition(String s) {

        List<List<String>> ans = new ArrayList<>();

        recur(0, s, ans, new ArrayList<>());

        return ans;

    }

    void recur(int index, String s, List<List<String>> ans, List<String> ds)
    {
        //base case
        if(index == s.length())
        {
            ans.add(new ArrayList<>(ds));
            return;
        }
        //startig from specific index
        for(int i = index; i < s.length(); i++)
        {
            //check if index to i is palindrome or not
            if(palindrome(s, index, i))
            {
                //if particular string is palindrome then add it
                ds.add(s.substring(index, i+1));
                //recur for next characters
                recur(i + 1, s, ans, ds);
                //remove last character from inner list
                ds.remove(ds.size() - 1);
            }
        }
    }
    boolean palindrome(String s, int index, int i)
    {
        while(index < i)
        {
            if(s.charAt(i) == s.charAt(index))
            {
                index++;
                i--;
            }
            else
            {
                return false;
            }
        }
        return true;
    }



//Day 9 (6. K-th permutation Sequence)  


public String getPermutation(int n, int k) {
        
        //this list will store starting value of n! permutations
        //i.e for n = 4, it will store 1 to 4 numbers which can be starting 
        //for all permutations
        ArrayList<Integer> block = new ArrayList<>();
        
        //this is for block size, i.e fixed or two then how many perm. remain.
        int fact = 1;
        
        //calculating factorial of n-1, as we have fixed first digit then left
        //permutations will be (n-1)!--> for n = 4, if we fix 1 as starting 
        //element then rem. perm. will be 3!-->{1,2,3,4},{1,2,4,3},{1,3,2,4}etc 
        for(int i = 1; i < n; i++)
        {
            fact = fact * i;
            //also added elements to list
            block.add(i);
        }
        //added nth element to list
        block.add(n);
        
        String ans ="";
        
        //for 0 based indexing
        k = k - 1;
        
        while(true)
        {
            //took element from from k/fact index from block of perm. and added to ans
            //k/fact to find the element from which kth perm will start 
            ans += block.get(k / fact);
            //now we have fixed one index and used one elemnt from n, so remove it also from block
            block.remove(k / fact);
            
            //iterate till all elements are not removed from list
            if(block.size() == 0)
                break;
            
            //also reduce size of k for n-1
            k = k % fact;
            // updated block size or no. of prem. remaining every time
            // 3! = 6 now next time --> 6/3 = 2 i.e 2!
            fact = fact / block.size();
        }
        return ans;
    }
    
