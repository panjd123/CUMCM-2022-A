#include <cmath>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>
#include <vector>
#define M_PI 3.14159265358979323846
#define eps 1e-7
using namespace std;
using pii = pair<int, int>;
using pdd = pair<double, double>;
const int N = 100;
pdd p[N];
int n;
double get_angle(pdd a, pdd b) {
    double angle1 = atan2(a.second, a.first), angle2 = atan2(b.second, b.first), angle = angle2 - angle1;
    if (angle < 0)
        angle += 2 * M_PI;
    return angle;
}
pdd operator-(pdd a, pdd b) {
    return {a.first - b.first, a.second - b.second};
}
bool check(vector<tuple<double, double, double>> v) {
    double mx = 0;
    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = 0; j < v.size(); j++) {
            if (i == j)
                continue;
            double x1, x2, x3;
            tie(x1, x2, x3) = v[i];
            double y1, y2, y3;
            tie(y1, y2, y3) = v[j];
            auto diff = fabs(x1 - y1) + fabs(x2 - y2) + fabs(x3 - y3);
            if (diff < eps) {
                return false;
            }
            mx = max(mx, diff);
        }
    }
    cout << mx << endl;
    return true;
}
int main() {
    n = 10 - 1;
    for (int i = 0; i < n; i++) {
        double theta = 2 * M_PI * i / n;
        p[i] = {cos(theta), sin(theta)};
    }
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            vector<tuple<double, double, double>> vis;
            for (int k = 0; k < n; k++) {
                if (i == k || j == k)
                    continue;
                pdd a = p[k] - p[i];
                pdd b = p[k] - p[j];
                pdd c = p[k];
                double angle1 = get_angle(a, b), angle2 = get_angle(a, c), angle3 = get_angle(b, c);
                if (angle1 > angle2)
                    swap(angle1, angle2);
                if (angle2 > angle3)
                    swap(angle2, angle3);
                if (angle1 > angle2)
                    swap(angle1, angle2);
                auto tag = make_tuple(angle1, angle2, angle3);
                vis.push_back(tag);
            }
            if (!check(vis)) {
                cout << i << " " << j << endl;
            }
        }
    }
}