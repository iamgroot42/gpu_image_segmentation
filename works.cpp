#include <cmath>
#include <vector>
#include <iostream>
#include <queue>
#include <stdio.h>

using namespace std;

typedef long long LL;

struct Edge {
  int from, to, cap, flow, index;
  Edge(int from, int to, int cap, int flow, int index) :
    from(from), to(to), cap(cap), flow(flow), index(index) {}
};

struct PushRelabel {
  int N;
  vector<vector<Edge> > G;
  vector<LL> excess;
  vector<int> dist, active, count;
  queue<int> Q;

  PushRelabel(int N) : N(N), G(N), excess(N), dist(N), active(N), count(2*N) {}

  void AddEdge(int from, int to, int cap) {
    G[from].push_back(Edge(from, to, cap, 0, G[to].size()));
    G[to].push_back(Edge(to, from, 0, 0, G[from].size() - 1));
  }

  void Enqueue(int v) { 
    if (!active[v] && excess[v] > 0) { active[v] = true; Q.push(v); } 
  }

  void Push(Edge &e) {
    int amt = int(min(excess[e.from], LL(e.cap - e.flow)));
    if (dist[e.from] <= dist[e.to] || amt == 0) return;
    e.flow += amt;
    G[e.to][e.index].flow -= amt;
    excess[e.to] += amt;    
    excess[e.from] -= amt;
    Enqueue(e.to);
  }
  
  void Gap(int k) {
    for (int v = 0; v < N; v++) {
      if (dist[v] < k) continue;
      count[dist[v]]--;
      dist[v] = max(dist[v], N+1);
      count[dist[v]]++;
      Enqueue(v);
    }
  }

  void Relabel(int v) {
    count[dist[v]]--;
    dist[v] = 2*N;
    for (int i = 0; i < G[v].size(); i++) 
      if (G[v][i].cap - G[v][i].flow > 0)
	dist[v] = min(dist[v], dist[G[v][i].to] + 1);
    count[dist[v]]++;
    Enqueue(v);
  }

  void Discharge(int v) {
    for (int i = 0; excess[v] > 0 && i < G[v].size(); i++) Push(G[v][i]);
    if (excess[v] > 0) {
      if (count[dist[v]] == 1) 
	Gap(dist[v]); 
      else
	Relabel(v);
    }
  }

  LL GetMaxFlow(int s, int t) {
    count[0] = N-1;
    count[N] = 1;
    dist[s] = N;
    active[s] = active[t] = true;
    for (int i = 0; i < G[s].size(); i++) {
      excess[s] += G[s][i].cap;
      Push(G[s][i]);
    }

    while (!Q.empty()) {
      int v = Q.front();
      Q.pop();
      active[v] = false;
      Discharge(v);
    }
    
    LL totflow = 0;
    for (int i = 0; i < G[s].size(); i++) totflow += G[s][i].flow;
    return totflow;
  }
};

int main() {
  // int n, m;
  // scanf("%d%d", &n, &m);

  // PushRelabel pr(n);
  // for (int i = 0; i < m; i++) { 
  //  int a, b, c;
  //   scanf("%d%d%d", &a, &b, &c);
  //   if (a == b) continue;
  //   pr.AddEdge(a-1, b-1, c);
  //   pr.AddEdge(b-1, a-1, c);
  // }
  // printf("%Ld\n", pr.GetMaxFlow(0, n-1));
  int i, m, n, x, y, z, count = 1;
  n = -1;
  while (n)
  {
    cin >> n;
    if (!n)
      break;
    PushRelabel pr(n);
    int source, sink;
    cin >> source >> sink >> m;
    // cout << source << ' ' << sink << '\n';
    while (m--)
    {
      cin >> x >> y >> z;
      if (x == y) continue;
      pr.AddEdge(x - 1, y - 1, z);
      pr.AddEdge(y - 1, x - 1, z);
    }
    cout << "Network " << count++ << '\n';
    cout << "The bandwidth is " << pr.GetMaxFlow(source-1, sink-1) << ".\n\n";
  }
  return 0;
}

