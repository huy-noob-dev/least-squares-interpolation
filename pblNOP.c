#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdarg.h>
#include <ctype.h>

// Cấu trúc dữ liệu động
typedef struct {
    double *x;
    double *y;
    int capacity;
    int size;
} Dataset;

// Khởi tạo dataset
void initDataset(Dataset *ds) {
    ds->x = ds->y = NULL;
    ds->capacity = ds->size = 0;
}

// Giải phóng bộ nhớ
void freeDataset(Dataset *ds) {
    if (ds->x) free(ds->x);
    if (ds->y) free(ds->y);
    ds->x = ds->y = NULL;
    ds->capacity = ds->size = 0;
}

// Mở rộng dung lượng khi cần
void expandDataset(Dataset *ds) {
    int new_capacity = ds->capacity == 0 ? 100 : ds->capacity * 2;
    double *new_x = (double*)realloc(ds->x, new_capacity * sizeof(double));
    double *new_y = (double*)realloc(ds->y, new_capacity * sizeof(double));
    
    if (!new_x || !new_y) {
        printf("Loi: Khong du bo nho!\n");
        free(new_x);
        free(new_y);
        exit(1);
    }
    
    ds->x = new_x;
    ds->y = new_y;
    ds->capacity = new_capacity;
}

// Thêm điểm dữ liệu
void addDataPoint(Dataset *ds, double x, double y) {
    if (ds->size >= ds->capacity) {
        expandDataset(ds);
    }
    ds->x[ds->size] = x;
    ds->y[ds->size] = y;
    ds->size++;
}

// Hàm phụ trợ
void clearInputBuffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF) {}
}

double safeDiv(double a, double b) {
    return (fabs(b) < DBL_EPSILON) ? 0.0 : (a / b);
}

// Hàm tính R²
double calculateR2(Dataset *ds, double (*model)(double, double[]), double coeff[], int degree) {
    double ss_tot = 0, ss_res = 0;
    double y_mean = 0;
    
    for (int i = 0; i < ds->size; i++) {
        y_mean += ds->y[i];
    }
    y_mean /= ds->size;
    
    for (int i = 0; i < ds->size; i++) {
        double y_pred = model(ds->x[i], coeff);
        ss_res += (ds->y[i] - y_pred) * (ds->y[i] - y_pred);
        ss_tot += (ds->y[i] - y_mean) * (ds->y[i] - y_mean);
    }
    
    return 1.0 - safeDiv(ss_res, ss_tot);
}

// Các hàm mô hình
double linearModel(double x, double coeff[]) {
    return coeff[0] + coeff[1] * x;
}

double logModel(double x, double coeff[]) {
    return coeff[0] + coeff[1] * log(x);
}

double expModel(double x, double coeff[]) {
    return coeff[0] * exp(coeff[1] * x);
}

double quadraticModel(double x, double coeff[]) {
    return coeff[0] + coeff[1] * x + coeff[2] * x * x;
}

double polyModel(double x, double coeff[]) {
    double result = 0;
    int degree = (int)coeff[0];
    for (int i = 0; i <= degree; i++) {
        result += coeff[i+1] * pow(x, i);
    }
    return result;
}

// Các hàm hồi quy
void linearRegression(Dataset *ds, FILE *logFile) {
    if (ds->size < 2) {
        printf("Can it nhat 2 diem de hoi quy tuyen tinh!\n");
        return;
    }

    double sum_x = 0, sum_y = 0, sum_x2 = 0, sum_xy = 0;
    
    printf("\n=== HOI QUY TUYEN TINH ===\n");
    printf("+-------+--------+--------+---------+----------+\n");
    printf("| %-5s | %-6s | %-6s | %-7s | %-8s |\n", "STT", "x", "y", "x^2", "x*y");
    printf("+-------+--------+--------+---------+----------+\n");
    
    for (int i = 0; i < ds->size; i++) {
        double xi = ds->x[i], yi = ds->y[i];
        double xi2 = xi * xi, xiyi = xi * yi;
        sum_x += xi;
        sum_y += yi;
        sum_x2 += xi2;
        sum_xy += xiyi;
        
        printf("| %-5d | %-6.2lf | %-6.2lf | %-7.2lf | %-8.2lf |\n", 
              i+1, xi, yi, xi2, xiyi);
        fprintf(logFile, "x[%d]=%.2lf y=%.2lf x^2=%.2lf x*y=%.2lf\n", 
               i, xi, yi, xi2, xiyi);
    }
    printf("+-------+--------+--------+---------+----------+\n");
    
    double denom = ds->size * sum_x2 - sum_x * sum_x;
    double a = safeDiv(sum_y * sum_x2 - sum_x * sum_xy, denom);
    double b = safeDiv(ds->size * sum_xy - sum_x * sum_y, denom);
    
    printf("\nPhuong trinh hoi quy:\n");
    printf("y = %.6lf + %.6lf * x\n", a, b);
    fprintf(logFile, "\n[Tuyen tinh] y = %.6lf + %.6lf * x\n", a, b);

    double coeff[2] = {a, b};
    double r2 = calculateR2(ds, linearModel, coeff, 1);
    printf("He so xac dinh R^2: %.6lf\n", r2);
    fprintf(logFile, "[Tuyen tinh] R^2 = %.6lf\n\n", r2);
}

void logRegression(Dataset *ds, FILE *logFile) {
    if (ds->size < 2) {
        printf("Can it nhat 2 diem de hoi quy logarit!\n");
        return;
    }

    // Kiểm tra dữ liệu hợp lệ
    for (int i = 0; i < ds->size; i++) {
        if (ds->x[i] <= 0) {
            printf("Loi: x[%d] = %.2lf <= 0 (khong hop le cho logarit)\n", i, ds->x[i]);
            return;
        }
    }

    double sum_lnx = 0, sum_y = 0, sum_lnx2 = 0, sum_lnx_y = 0;
    
    printf("\n=== HOI QUY LOGARIT ===\n");
    printf("+-------+--------+--------+---------+-----------+-------------+\n");
    printf("| %-5s | %-6s | %-6s | %-7s | %-9s | %-11s |\n", 
           "STT", "x", "y", "ln(x)", "ln(x)^2", "ln(x)*y");
    printf("+-------+--------+--------+---------+-----------+-------------+\n");
    
    for (int i = 0; i < ds->size; i++) {
        double lnx = log(ds->x[i]);
        sum_lnx += lnx;
        sum_y += ds->y[i];
        sum_lnx2 += lnx * lnx;
        sum_lnx_y += lnx * ds->y[i];
        
        printf("| %-5d | %-6.2lf | %-6.2lf | %-7.3lf | %-9.3lf | %-11.3lf |\n",
              i+1, ds->x[i], ds->y[i], lnx, lnx * lnx, lnx * ds->y[i]);
        fprintf(logFile, "x[%d]=%.2lf ln(x)=%.3lf y=%.2lf ln(x)^2=%.3lf ln(x)*y=%.3lf\n",
               i, ds->x[i], lnx, ds->y[i], lnx * lnx, lnx * ds->y[i]);
    }
    printf("+-------+--------+--------+---------+-----------+-------------+\n");
    
    double denom = ds->size * sum_lnx2 - sum_lnx * sum_lnx;
    double a = safeDiv(sum_y * sum_lnx2 - sum_lnx * sum_lnx_y, denom);
    double b = safeDiv(ds->size * sum_lnx_y - sum_lnx * sum_y, denom);
    
    printf("\nPhuong trinh hoi quy:\n");
    printf("y = %.6lf + %.6lf * ln(x)\n", a, b);
    fprintf(logFile, "\n[Logarit] y = %.6lf + %.6lf * ln(x)\n", a, b);

    double coeff[2] = {a, b};
    double r2 = calculateR2(ds, logModel, coeff, 1);
    printf("He so xac dinh R^2: %.6lf\n", r2);
    fprintf(logFile, "[Logarit] R^2 = %.6lf\n\n", r2);
}

void exponentialRegression(Dataset *ds, FILE *logFile) {
    if (ds->size < 2) {
        printf("Can it nhat 2 diem de hoi quy ham mu!\n");
        return;
    }

    // Kiểm tra dữ liệu hợp lệ
    for (int i = 0; i < ds->size; i++) {
        if (ds->y[i] <= 0) {
            printf("Loi: y[%d] = %.2lf <= 0 (khong hop le cho ham mu)\n", i, ds->y[i]);
            return;
        }
    }

    double sum_x = 0, sum_lny = 0, sum_x2 = 0, sum_x_lny = 0;
    
    printf("\n=== HOI QUY HAM MU ===\n");
    printf("+-------+--------+--------+---------+-----------+\n");
    printf("| %-5s | %-6s | %-6s | %-7s | %-9s |\n", 
           "STT", "x", "y", "ln(y)", "x*ln(y)");
    printf("+-------+--------+--------+---------+-----------+\n");
    
    for (int i = 0; i < ds->size; i++) {
        double lny = log(ds->y[i]);
        sum_x += ds->x[i];
        sum_lny += lny;
        sum_x2 += ds->x[i] * ds->x[i];
        sum_x_lny += ds->x[i] * lny;
        
        printf("| %-5d | %-6.2lf | %-6.2lf | %-7.3lf | %-9.3lf |\n", 
              i+1, ds->x[i], ds->y[i], lny, ds->x[i]*lny);
        fprintf(logFile, "x[%d]=%.2lf y=%.2lf ln(y)=%.3lf x*ln(y)=%.3lf\n", 
               i, ds->x[i], ds->y[i], lny, ds->x[i]*lny);
    }
    printf("+-------+--------+--------+---------+-----------+\n");
    
    double denom = ds->size * sum_x2 - sum_x * sum_x;
    double A = safeDiv(sum_lny * sum_x2 - sum_x * sum_x_lny, denom);
    double B = safeDiv(ds->size * sum_x_lny - sum_x * sum_lny, denom);
    double a = exp(A), b = B;
    
    printf("\nPhuong trinh hoi quy:\n");
    printf("y = %.6lf * e^(%.6lf * x)\n", a, b);
    fprintf(logFile, "\n[Ham mu] y = %.6lf * e^(%.6lf * x)\n", a, b);

    double coeff[2] = {a, b};
    double r2 = calculateR2(ds, expModel, coeff, 1);
    printf("He so xac dinh R^2: %.6lf\n", r2);
    fprintf(logFile, "[Ham mu] R^2 = %.6lf\n\n", r2);
}

void quadraticRegression(Dataset *ds, FILE *logFile) {
    if (ds->size < 3) {
        printf("Can it nhat 3 diem de hoi quy bac hai!\n");
        return;
    }

    double sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;
    
    printf("\n=== HOI QUY BAC HAI ===\n");
    printf("+-------+--------+--------+--------+--------+--------+--------+---------+\n");
    printf("| %-5s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s | %-7s |\n", 
           "STT", "x", "y", "x^2", "x^3", "x^4", "x*y", "x^2*y");
    printf("+-------+--------+--------+--------+--------+--------+--------+---------+\n");
    
    for (int i = 0; i < ds->size; i++) {
        double xi = ds->x[i], yi = ds->y[i];
        double xi2 = xi * xi, xi3 = xi2 * xi, xi4 = xi2 * xi2;
        sx += xi; sx2 += xi2; sx3 += xi3; sx4 += xi4;
        sy += yi; sxy += xi * yi; sx2y += xi2 * yi;
        
        printf("| %-5d | %-6.2lf | %-6.2lf | %-6.2lf | %-6.2lf | %-6.2lf | %-6.2lf | %-7.2lf |\n",
              i+1, xi, yi, xi2, xi3, xi4, xi * yi, xi2 * yi);
        fprintf(logFile, "x=%.2lf y=%.2lf x^2=%.2lf x^3=%.2lf x^4=%.2lf x*y=%.2lf x^2*y=%.2lf\n",
               xi, yi, xi2, xi3, xi4, xi * yi, xi2 * yi);
    }
    printf("+-------+--------+--------+--------+--------+--------+--------+---------+\n");
    
    // Giải hệ phương trình
    double A[3][4] = {
        {(double)ds->size, sx, sx2, sy},
        {sx, sx2, sx3, sxy},
        {sx2, sx3, sx4, sx2y}
    };
    
    // Phương pháp khử Gauss
    for (int k = 0; k < 2; k++) {
        for (int i = k+1; i < 3; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < 4; j++) {
                A[i][j] -= factor * A[k][j];
            }
        }
    }
    
    // Thế ngược
    double c = A[2][3] / A[2][2];
    double b = (A[1][3] - A[1][2]*c) / A[1][1];
    double a = (A[0][3] - A[0][2]*c - A[0][1]*b) / A[0][0];
    
    printf("\nPhuong trinh hoi quy:\n");
    printf("y = %.6lf + %.6lf * x + %.6lf * x^2\n", a, b, c);
    fprintf(logFile, "\n[Bac hai] y = %.6lf + %.6lf * x + %.6lf * x^2\n", a, b, c);

    double coeff[3] = {a, b, c};
    double r2 = calculateR2(ds, quadraticModel, coeff, 2);
    printf("He so xac dinh R^2: %.6lf\n", r2);
    fprintf(logFile, "[Bac hai] R^2 = %.6lf\n\n", r2);
}

void polyRegression(Dataset *ds, int degree, FILE *logFile) {
    if (ds->size <= degree) {
        printf("So diem du lieu phai lon hon bac da thuc!\n");
        return;
    }

    int n = degree + 1;
    double X[2*degree+1], Y[degree+1];
    double A[degree+1][degree+2];
    
    // Khởi tạo ma trận
    memset(X, 0, sizeof(X));
    memset(Y, 0, sizeof(Y));
    
    // Tính các tổng lũy thừa
    for (int i = 0; i <= 2*degree; i++) {
        for (int j = 0; j < ds->size; j++) {
            X[i] += pow(ds->x[j], i);
        }
    }
    
    for (int i = 0; i <= degree; i++) {
        for (int j = 0; j < ds->size; j++) {
            Y[i] += pow(ds->x[j], i) * ds->y[j];
        }
    }
    
    // Xây dựng ma trận hệ số
    for (int i = 0; i <= degree; i++) {
        for (int j = 0; j <= degree; j++) {
            A[i][j] = X[i+j];
        }
        A[i][degree+1] = Y[i];
    }
    
    // Giải hệ phương trình bằng phép khử Gauss
    for (int k = 0; k <= degree; k++) {
        // Tìm hàng có phần tử lớn nhất
        int max_row = k;
        for (int i = k+1; i <= degree; i++) {
            if (fabs(A[i][k]) > fabs(A[max_row][k])) {
                max_row = i;
            }
        }
        
        // Đổi hàng
        if (max_row != k) {
            for (int j = k; j <= degree+1; j++) {
                double temp = A[k][j];
                A[k][j] = A[max_row][j];
                A[max_row][j] = temp;
            }
        }
        
        // Khử
        for (int i = k+1; i <= degree; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j <= degree+1; j++) {
                A[i][j] -= factor * A[k][j];
            }
        }
    }
    
    // Thế ngược
    double coeff[degree+2];
    coeff[0] = degree; // Lưu bậc đa thức
    
    for (int i = degree; i >= 0; i--) {
        coeff[i+1] = A[i][degree+1];
        for (int j = i+1; j <= degree; j++) {
            coeff[i+1] -= A[i][j] * coeff[j+1];
        }
        coeff[i+1] /= A[i][i];
    }
    
    printf("\nPhuong trinh hoi quy da thuc bac %d:\n", degree);
    printf("y = ");
    for (int i = 0; i <= degree; i++) {
        if (i == 0) {
            printf("%.6lf", coeff[1]);
        } else {
            printf(" %+.6lfx^%d", coeff[i+1], i);
        }
    }
    printf("\n");
    
    fprintf(logFile, "\n[Da thuc bac %d] y = ", degree);
    for (int i = 0; i <= degree; i++) {
        fprintf(logFile, "%+.6lfx^%d ", coeff[i+1], i);
    }
    fprintf(logFile, "\n");

    double r2 = calculateR2(ds, polyModel, coeff, degree);
    printf("He so xac dinh R^2: %.6lf\n", r2);
    fprintf(logFile, "[Da thuc bac %d] R^2 = %.6lf\n\n", degree, r2);
}

// Hiển thị menu
void displayMainMenu() {
    printf("\n+-------------------------------------+\n");
    printf("|    CHUONG TRINH HOI QUY DU LIEU    |\n");
    printf("+-------------------------------------+\n");
    printf("| 1. Hoi quy tuyen tinh              |\n");
    printf("| 2. Hoi quy logarit                 |\n");
    printf("| 3. Hoi quy ham mu                  |\n");
    printf("| 4. Hoi quy bac hai                 |\n");
    printf("| 5. Hoi quy da thuc bac n           |\n");
    printf("| 6. Xem lai du lieu nhap            |\n");
    printf("| 0. Thoat                           |\n");
    printf("+-------------------------------------+\n");
    printf("Lua chon cua ban: ");
}

void displayDataInputMenu() {
    printf("\n+-------------------------------------+\n");
    printf("|       CHON PHUONG THUC NHAP        |\n");
    printf("+-------------------------------------+\n");
    printf("| 1. Nhap tu ban phim                |\n");
    printf("| 2. Nhap tu file                    |\n");
    printf("| 0. Quay lai                        |\n");
    printf("+-------------------------------------+\n");
    printf("Lua chon cua ban: ");
}

void displayInfoPanel() {
    printf("\n");
    printf("   _____     ____     _      \n");
    printf("  |  __ \\   |  _ \\   | |     \n");
    printf("  | |__) |  | |_) |  | |     \n");
    printf("  |  ___/   |  _ <   | |     \n");
    printf("  | |       | |_) |  | |____ \n");
    printf("  |_|       |____/   |______|\n");
    printf("\n");
    printf("DU AN LAP TRINH TINH TOAN\n");
    printf("DE TAI: NOI SUY BINH PHUONG TOI THIEU\n\n");
    printf("HO TEN SINH VIEN:\n");
    printf("- Nguyen Duc Huy        - 102240310\n");
    printf("- Hoang Duc Quyen      - 102240217\n");
    printf("- Pham Ngoc Truong Son - 102240218\n");
    printf("\n");
}

void displayInputData(Dataset *ds) {
    printf("                       \n+-------------------------------------+\n");
    printf("|           DU LIEU DA NHAP           |\n");
    printf("+-------------------------------------+\n");
    printf("| %-5s | %-10s | %-10s |\n", "STT", "x", "y");
    printf("+-------+------------+------------+\n");
    
    int displayLimit = ds->size > 20 ? 20 : ds->size;
    for (int i = 0; i < displayLimit; i++) {
        printf("| %-5d | %-10.2lf | %-10.2lf |\n", i+1, ds->x[i], ds->y[i]);
    }
    
    if (ds->size > 20) {
        printf("| %-5s | %-10s | %-10s |\n", "...", "...", "...");
        printf("| %-5d | %-10.2lf | %-10.2lf |\n", ds->size, ds->x[ds->size-1], ds->y[ds->size-1]);
    }
    printf("+-------+------------+------------+\n");
    printf("Tong cong: %d diem du lieu\n", ds->size);
}

int main() {
    Dataset data;
    initDataset(&data);
    FILE *logFile = fopen("regression_log.txt", "w");
    
    displayInfoPanel();

    // Nhập dữ liệu
    int dataChoice;
    do {
        displayDataInputMenu();
        if (scanf("%d", &dataChoice)) {
            clearInputBuffer();
            
            switch (dataChoice) {
                case 1: {
                    printf("Nhap so diem: ");
                    int n;
                    if (scanf("%d", &n) != 1 || n <= 0) {
                        printf("So diem khong hop le!\n");
                        clearInputBuffer();
                        continue;
                    }
                    clearInputBuffer();
                    
                    printf("Nhap cac cap (x y), moi cap tren 1 dong:\n");
                    for (int i = 0; i < n; i++) {
                        double x, y;
                        printf("Diem %d: ", i+1);
                        if (scanf("%lf %lf", &x, &y) != 2) {
                            printf("Nhap khong hop le! Vui long nhap lai.\n");
                            clearInputBuffer();
                            i--; // Nhập lại điểm này
                            continue;
                        }
                        addDataPoint(&data, x, y);
                    }
                    break;
                }
                case 2: {
                    char filename[256];
                    printf("Nhap ten file: ");
                    if (scanf("%255s", filename) != 1) {
                        printf("Ten file khong hop le!\n");
                        clearInputBuffer();
                        continue;
                    }
                    clearInputBuffer();
                    
                    FILE *f = fopen(filename, "r");
                    if (!f) {
                        perror("Loi mo file");
                        continue;
                    }
                    
                    double x, y;
                    while (fscanf(f, "%lf %lf", &x, &y) == 2) {
                        addDataPoint(&data, x, y);
                        if (data.size % 1000 == 0) {
                            printf("Da doc %d diem...\n", data.size);
                        }
                    }
                    fclose(f);
                    printf("Da doc duoc %d diem tu file.\n", data.size);
                    break;
                }
                case 0:
                    break;
                default:
                    printf("Lua chon khong hop le!\n");
            }
        } else {
            printf("Vui long nhap so!\n");
            clearInputBuffer();
        }
    } while (dataChoice != 0 && data.size == 0);

    if (data.size == 0) {
        printf("Khong co du lieu de xu ly.\n");
        freeDataset(&data);
        fclose(logFile);
        return 0;
    }

    // Menu chính
    int choice;
    do {
        displayMainMenu();
        if (scanf("%d", &choice)) {
            clearInputBuffer();
            
            switch (choice) {
                case 1:
                    linearRegression(&data, logFile);
                    break;
                case 2:
                    logRegression(&data, logFile);
                    break;
                case 3:
                    exponentialRegression(&data, logFile);
                    break;
                case 4:
                    quadraticRegression(&data, logFile);
                    break;
                case 5: {
                    printf("Nhap bac da thuc (toi da %d): ", data.size-1);
                    int degree;
                    if (scanf("%d", &degree) != 1 || degree < 1 || degree > data.size-1) {
                        printf("Bac da thuc khong hop le!\n");
                        clearInputBuffer();
                        break;
                    }
                    polyRegression(&data, degree, logFile);
                    break;
                }
                case 6:
                    displayInputData(&data);
                    break;
                case 0:
                    printf("Tam biet!\n");
                    break;
                default:
                    printf("Lua chon khong hop le!\n");
            }
        } else {
            printf("Vui long nhap so tu 0 den 6!\n");
            clearInputBuffer();
        }
    } while (choice != 0);

    freeDataset(&data);
    fclose(logFile);
    return 0;
}
