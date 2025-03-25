import sys

def main():
    q = int(sys.stdin.readline())
    for _ in range(q):
        n, m, k = map(int, sys.stdin.readline().split())
        total = n + m
        if total < k:
            print(-1)
            continue
        d_min = max(0, total - k)
        d_max = min(k, total)
        required_parity = (k - total) % 2
        d_candidate = -1
        # Iterate from d_max down to d_min to find the largest d with correct parity
        for d in range(d_max, d_min - 1, -1):
            if d % 2 == required_parity:
                d_candidate = d
                break
        print(d_candidate if d_candidate != -1 else -1)

if __name__ == "__main__":
    main()


