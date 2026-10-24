// lm_nn.cpp -- Levenberg-Marquardt vs SGD training a tiny 8-H-1 tanh network.
// No external libraries. Build:  g++ -O2 -std=c++17 -o lm_nn lm_nn.cpp
//
// Usage:
//   ./lm_nn gen    out.csv                       synthetic 1030x9 file (ONLY to test the code)
//   ./lm_nn check  data.csv                      analytic Jacobian vs finite differences
//   ./lm_nn train  data.csv [seed lmIters sgdEpochs lr batch H]
//   ./lm_nn scale  data.csv [maxH]               cost of one LM iteration vs one SGD epoch for several H
//
// Model: f(x) = b2 + sum_j W2_j * tanh(b1_j + sum_k W1_jk x_k)
// theta layout: [W1 (H*in, row-major) | b1 (H) | W2 (H) | b2 (1)],  n = H*in + 2H + 1  (81 for H=8, in=8)
// Residual: r_i = f(x_i) - y_i.   S(theta) = sum_i r_i^2.
// LM step:  (J^T J + lambda*diag(J^T J)) d = -J^T r        (Marquardt scaling)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
using Clock = chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return chrono::duration<double>(b - a).count(); }

struct Dataset {
    vector<vector<double>> X;
    vector<double> y;
    int m() const { return (int)y.size(); }
};

static bool parseLine(string s, vector<double>& v) {
    bool semi = s.find(';') != string::npos;  // Spanish-locale Excel: ';' separator, ',' decimal
    for (char& c : s) {
        if (c == '"' || c == '\r') c = ' ';
        else if (semi) { if (c == ',') c = '.'; else if (c == ';') c = ' '; }
        else if (c == ',' || c == '\t') c = ' ';
    }
    istringstream is(s);
    v.clear();
    string tok;
    while (is >> tok) {
        char* e;
        double d = strtod(tok.c_str(), &e);
        if (*e != '\0') return false;
        v.push_back(d);
    }
    return v.size() == 9;
}

static void loadData(const string& path, Dataset& tr, Dataset& te, double& ymean, double& ystd) {
    ifstream f(path);
    if (!f) { cerr << "No puedo abrir " << path << "\n"; exit(1); }
    vector<vector<double>> rows;
    string line;
    vector<double> v;
    while (getline(f, line))
        if (parseLine(line, v)) rows.push_back(v);
    if (rows.empty()) { cerr << "No se leyeron filas validas (9 columnas numericas)\n"; exit(1); }
    mt19937 rng(12345);
    shuffle(rows.begin(), rows.end(), rng);
    int M = (int)rows.size(), mtr = (int)(0.8 * M);
    vector<double> mu(9, 0.0), sd(9, 0.0);
    for (int i = 0; i < mtr; i++) for (int c = 0; c < 9; c++) mu[c] += rows[i][c];
    for (int c = 0; c < 9; c++) mu[c] /= mtr;
    for (int i = 0; i < mtr; i++) for (int c = 0; c < 9; c++) sd[c] += (rows[i][c] - mu[c]) * (rows[i][c] - mu[c]);
    for (int c = 0; c < 9; c++) { sd[c] = sqrt(sd[c] / mtr); if (sd[c] < 1e-12) sd[c] = 1; }
    ymean = mu[8]; ystd = sd[8];
    for (int i = 0; i < M; i++) {
        vector<double> x(8);
        for (int c = 0; c < 8; c++) x[c] = (rows[i][c] - mu[c]) / sd[c];
        double y = (rows[i][8] - mu[8]) / sd[8];
        Dataset& d = (i < mtr) ? tr : te;
        d.X.push_back(x);
        d.y.push_back(y);
    }
    printf("Datos: %d filas (train %d, test %d)\n", M, tr.m(), te.m());
}


struct Net {
    int in, H, n;
    Net(int in_, int H_) : in(in_), H(H_), n(H_ * in_ + 2 * H_ + 1) {}
    int ob1() const { return H * in; }
    int oW2() const { return H * in + H; }
    int ob2() const { return H * in + 2 * H; }

    double forward(const vector<double>& th, const vector<double>& x, vector<double>& a) const {
        a.resize(H);
        double out = th[ob2()];
        for (int j = 0; j < H; j++) {
            double z = th[ob1() + j];
            for (int k = 0; k < in; k++) z += th[j * in + k] * x[k];
            a[j] = tanh(z);
            out += th[oW2() + j] * a[j];
        }
        return out;
    }
    // One row of J: d f / d theta  (r = f - y, so dr/dtheta = df/dtheta)
    void jacRow(const vector<double>& th, const vector<double>& x, const vector<double>& a, double* row) const {
        for (int j = 0; j < H; j++) {
            double d = th[oW2() + j] * (1.0 - a[j] * a[j]);  // W2_j * tanh'(z_j)
            for (int k = 0; k < in; k++) row[j * in + k] = d * x[k];
            row[ob1() + j] = d;
            row[oW2() + j] = a[j];
        }
        row[ob2()] = 1.0;
    }
    vector<double> init(unsigned seed) const {
        mt19937 rng(seed);
        vector<double> th(n, 0.0);
        normal_distribution<double> g(0.0, 1.0);
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < in; k++) th[j * in + k] = g(rng) / sqrt((double)in);
            th[oW2() + j] = g(rng) / sqrt((double)H);
        }
        return th;
    }
};

static double sse(const Net& net, const vector<double>& th, const Dataset& d) {
    vector<double> a;
    double s = 0;
    for (int i = 0; i < d.m(); i++) { double r = net.forward(th, d.X[i], a) - d.y[i]; s += r * r; }
    return s;
}
static double mse(const Net& net, const vector<double>& th, const Dataset& d) { return sse(net, th, d) / d.m(); }

static bool cholSolve(vector<double>& A, int n, vector<double>& b) {
    for (int j = 0; j < n; j++) {
        double s = A[j * n + j];
        for (int k = 0; k < j; k++) s -= A[j * n + k] * A[j * n + k];
        if (s <= 0) return false;
        double ljj = sqrt(s);
        A[j * n + j] = ljj;
        for (int i = j + 1; i < n; i++) {
            double t = A[i * n + j];
            for (int k = 0; k < j; k++) t -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = t / ljj;
        }
    }
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= A[i * n + k] * b[k];
        b[i] = s / A[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < n; k++) s -= A[k * n + i] * b[k];
        b[i] = s / A[i * n + i];
    }
    return true;
}


static double buildNormalEq(const Net& net, const vector<double>& th, const Dataset& d, vector<double>& JtJ, vector<double>& g) {
    int n = net.n;
    fill(JtJ.begin(), JtJ.end(), 0.0);
    fill(g.begin(), g.end(), 0.0);
    vector<double> a(net.H), row(n);
    double S = 0;
    for (int i = 0; i < d.m(); i++) {
        double r = net.forward(th, d.X[i], a) - d.y[i];
        net.jacRow(th, d.X[i], a, row.data());
        S += r * r;
        for (int p = 0; p < n; p++) {
            double rp = row[p];
            g[p] += rp * r;
            double* dst = &JtJ[p * n];
            for (int q = p; q < n; q++) dst[q] += rp * row[q];
        }
    }
    for (int p = 0; p < n; p++) for (int q = 0; q < p; q++) JtJ[p * n + q] = JtJ[q * n + p];
    return S;
}

//  Levenberg-Marquardt
static void runLM(const Net& net, vector<double> th, const Dataset& tr, const Dataset& te, int maxIter, const string& logPath, double ystd) {
    int n = net.n;
    vector<double> JtJ((size_t)n * n), A((size_t)n * n), g(n), d(n), thn(n);
    double lambda = 1e-2, tTrain = 0;
    int rejTotal = 0;
    ofstream f(logPath);
    f << "iter,time_s,train_mse,test_mse,lambda,rejected_total\n";
    f << 0 << "," << 0 << "," << mse(net, th, tr) << "," << mse(net, th, te) << "," << lambda << ",0\n";
    int it = 1;
    for (; it <= maxIter; it++) {
        auto t0 = Clock::now();
        double S = buildNormalEq(net, th, tr, JtJ, g);
        bool accepted = false;
        double Snew = S;
        for (int tries = 0; tries < 40; tries++) {
            A = JtJ;
            for (int p = 0; p < n; p++) A[p * n + p] += lambda * max(JtJ[p * n + p], 1e-9);
            for (int p = 0; p < n; p++) d[p] = -g[p];
            if (!cholSolve(A, n, d)) { lambda *= 4; rejTotal++; continue; }
            for (int p = 0; p < n; p++) thn[p] = th[p] + d[p];
            Snew = sse(net, thn, tr);
            if (Snew < S) { th = thn; lambda = max(lambda / 3.0, 1e-12); accepted = true; break; }
            lambda *= 4; rejTotal++;
        }
        tTrain += secs(t0, Clock::now());
        if (!accepted) { printf("LM: sin mejora posible en iter %d, termina.\n", it); break; }
        f << it << "," << tTrain << "," << Snew / tr.m() << "," << mse(net, th, te) << "," << lambda << "," << rejTotal << "\n";
        if ((S - Snew) / S < 1e-10) { printf("LM: convergio (mejora relativa < 1e-10) en iter %d.\n", it); break; }
    }
    printf("LM   : iters=%d  tiempo=%.3fs  train RMSE=%.3f MPa  test RMSE=%.3f MPa  rechazos=%d\n", min(it, maxIter), tTrain,
           sqrt(mse(net, th, tr)) * ystd, sqrt(mse(net, th, te)) * ystd, rejTotal);
}

static void runSGD(const Net& net, vector<double> th, const Dataset& tr, const Dataset& te, int epochs, double lr, int batch,
                   unsigned seed, const string& logPath, double ystd) {
    int n = net.n, m = tr.m();
    vector<int> idx(m);
    iota(idx.begin(), idx.end(), 0);
    mt19937 rng(seed + 777);
    vector<double> a(net.H), row(n), grad(n);
    double tTrain = 0;
    ofstream f(logPath);
    f << "epoch,time_s,train_mse,test_mse\n";
    f << 0 << "," << 0 << "," << mse(net, th, tr) << "," << mse(net, th, te) << "\n";
    for (int ep = 1; ep <= epochs; ep++) {
        auto t0 = Clock::now();
        shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < m; s += batch) {
            int e = min(m, s + batch), B = e - s;
            fill(grad.begin(), grad.end(), 0.0);
            for (int q = s; q < e; q++) {
                int i = idx[q];
                double r = net.forward(th, tr.X[i], a) - tr.y[i];
                net.jacRow(th, tr.X[i], a, row.data());
                for (int p = 0; p < n; p++) grad[p] += 2.0 * r * row[p] / B;
            }
            for (int p = 0; p < n; p++) th[p] -= lr * grad[p];
        }
        tTrain += secs(t0, Clock::now());
        f << ep << "," << tTrain << "," << mse(net, th, tr) << "," << mse(net, th, te) << "\n";
    }
    printf("SGD  : epochs=%d lr=%g batch=%d  tiempo=%.3fs  train RMSE=%.3f MPa  test RMSE=%.3f MPa\n", epochs, lr, batch, tTrain,
           sqrt(mse(net, th, tr)) * ystd, sqrt(mse(net, th, te)) * ystd);
}

static void modeGen(const string& out) {
    mt19937 rng(1);
    uniform_real_distribution<double> U(0, 1);
    normal_distribution<double> N(0, 1);
    Net teacher(8, 5);
    vector<double> th = teacher.init(99);
    ofstream f(out);
    f << "c1,c2,c3,c4,c5,c6,c7,c8,y\n";
    vector<double> a;
    for (int i = 0; i < 1030; i++) {
        vector<double> x(8);
        for (int k = 0; k < 8; k++) x[k] = U(rng) * 2 - 1;
        double y = 35 + 12 * teacher.forward(th, x, a) + 1.5 * N(rng);
        for (int k = 0; k < 8; k++) f << x[k] * (k + 1) * 50 << ",";
        f << y << "\n";
    }
}

static void modeCheck(const string& path) {
    Dataset tr, te;
    double ym, ys;
    loadData(path, tr, te, ym, ys);
    for (int H : {2, 8}) {
        Net net(8, H);
        vector<double> th = net.init(5), a(H), row(net.n);
        mt19937 rng(3);
        normal_distribution<double> g(0, 0.5);
        for (double& v : th) v += g(rng);  // make biases / b2 non-zero as well
        double worst = 0;
        for (int i = 0; i < 20; i++) {
            net.forward(th, tr.X[i], a);
            net.jacRow(th, tr.X[i], a, row.data());
            for (int p = 0; p < net.n; p++) {
                double eps = 1e-6, o = th[p];
                vector<double> aa;
                th[p] = o + eps; double fp = net.forward(th, tr.X[i], aa);
                th[p] = o - eps; double fm = net.forward(th, tr.X[i], aa);
                th[p] = o;
                double num = (fp - fm) / (2 * eps);
                worst = max(worst, fabs(num - row[p]) / max(1.0, fabs(num)));
            }
        }
        printf("Jacobiano H=%d (n=%d): max error relativo vs diferencias finitas = %.2e  %s\n", H, net.n, worst, worst < 1e-6 ? "OK" : "FALLA");
    }
}

static void modeScale(const string& path, int maxH) {
    Dataset tr, te;
    double ym, ys;
    loadData(path, tr, te, ym, ys);
    printf("H,n_params,lm_iter_s,sgd_epoch_s\n");
    for (int H = 2; H <= maxH; H *= 2) {
        Net net(8, H);
        int n = net.n;
        vector<double> th = net.init(1), JtJ((size_t)n * n), A((size_t)n * n), g(n), d(n);
        int reps = n < 500 ? 3 : 1;
        auto t0 = Clock::now();
        for (int r = 0; r < reps; r++) {
            buildNormalEq(net, th, tr, JtJ, g);
            A = JtJ;
            for (int p = 0; p < n; p++) { A[p * n + p] += 1e-2 * max(JtJ[p * n + p], 1e-9); d[p] = -g[p]; }
            cholSolve(A, n, d);
        }
        double tLM = secs(t0, Clock::now()) / reps;
        vector<double> a(H), row(n), grad(n);
        t0 = Clock::now();
        for (int s = 0; s < tr.m(); s += 32) {
            int e = min(tr.m(), s + 32);
            fill(grad.begin(), grad.end(), 0.0);
            for (int i = s; i < e; i++) {
                double r = net.forward(th, tr.X[i], a) - tr.y[i];
                net.jacRow(th, tr.X[i], a, row.data());
                for (int p = 0; p < n; p++) grad[p] += 2.0 * r * row[p] / (e - s);
            }
            for (int p = 0; p < n; p++) th[p] -= 1e-3 * grad[p];
        }
        double tS = secs(t0, Clock::now());
        printf("%d,%d,%.5f,%.5f\n", H, n, tLM, tS);
        fflush(stdout);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { cerr << "Uso: ver comentarios al inicio del archivo\n"; return 1; }
    string mode = argv[1], path = argv[2];
    if (mode == "gen") { modeGen(path); return 0; }
    if (mode == "check") { modeCheck(path); return 0; }
    if (mode == "scale") { modeScale(path, argc > 3 ? atoi(argv[3]) : 128); return 0; }
    if (mode == "train") {
        unsigned seed = argc > 3 ? atoi(argv[3]) : 1;
        int lmIters = argc > 4 ? atoi(argv[4]) : 200;
        int epochs = argc > 5 ? atoi(argv[5]) : 2000;
        double lr = argc > 6 ? atof(argv[6]) : 0.01;
        int batch = argc > 7 ? atoi(argv[7]) : 32;
        int H = argc > 8 ? atoi(argv[8]) : 8;
        Dataset tr, te;
        double ym, ys;
        loadData(path, tr, te, ym, ys);
        Net net(8, H);
        printf("Red 8-%d-1, %d parametros, seed=%u\n", H, net.n, seed);
        vector<double> th0 = net.init(seed);  // same init for both optimizers
        runLM(net, th0, tr, te, lmIters, "lm_log.csv", ys);
        runSGD(net, th0, tr, te, epochs, lr, batch, seed, "sgd_log.csv", ys);
        return 0;
    }
    cerr << "Modo desconocido\n";
    return 1;
}
