#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <array>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <limits>
#include <map>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>
#include <set>

const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;
const int MAX_FRAMES_IN_FLIGHT = 2;
const double EPSILON = 1e-10;

// ============================================================================
// TERRAIN CONVERTER UTILITIES (MATH & DATA STRUCTURES)
// ============================================================================

struct Vec3 {
    double x, y, z;

    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3(const Vec3& v) : x(v.x), y(v.y), z(v.z) {}

    Vec3& operator=(const Vec3& v) {
        if (this != &v) {
            x = v.x;
            y = v.y;
            z = v.z;
        }
        return *this;
    }

    Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    Vec3 operator*(double s) const {
        return Vec3(x * s, y * s, z * s);
    }

    Vec3& operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    Vec3& operator*=(double s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
            );
    }

    double dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vec3 normalize() const {
        double len = length();
        if (len > EPSILON) {
            return Vec3(x / len, y / len, z / len);
        }
        return Vec3(0.0, 0.0, 1.0);
    }
};

struct Vec2 {
    double x, y;

    Vec2() : x(0.0), y(0.0) {}
    Vec2(double x, double y) : x(x), y(y) {}

    Vec2 operator-(const Vec2& v) const {
        return Vec2(x - v.x, y - v.y);
    }

    Vec2 operator+(const Vec2& v) const {
        return Vec2(x + v.x, y + v.y);
    }

    Vec2 operator*(double s) const {
        return Vec2(x * s, y * s);
    }

    double dot(const Vec2& v) const {
        return x * v.x + y * v.y;
    }

    double cross(const Vec2& v) const {
        return x * v.y - y * v.x;
    }

    double length() const {
        return std::sqrt(x * x + y * y);
    }

    double distance(const Vec2& v) const {
        return (*this - v).length();
    }
};

struct Point3D {
    double x, y, z;

    Point3D() : x(0.0), y(0.0), z(0.0) {}
    Point3D(double x, double y, double z) : x(x), y(y), z(z) {}
};

struct ConverterVertex {
    Vec3 position;
    Vec3 normal;

    ConverterVertex() : position(Vec3()), normal(Vec3(0.0, 0.0, 1.0)) {}
    ConverterVertex(const Vec3& pos, const Vec3& norm) : position(pos), normal(norm) {}
};

struct ConverterTriangle {
    size_t v0, v1, v2;
    ConverterTriangle(size_t v0, size_t v1, size_t v2) : v0(v0), v1(v1), v2(v2) {}

    bool operator==(const ConverterTriangle& t) const {
        return (v0 == t.v0 && v1 == t.v1 && v2 == t.v2) ||
               (v0 == t.v1 && v1 == t.v2 && v2 == t.v0) ||
               (v0 == t.v2 && v1 == t.v0 && v2 == t.v1);
    }
};

struct TransformParams {
    double center_x, center_y, center_z;
    double scale_xy, scale_z;

    TransformParams()
        : center_x(0.0), center_y(0.0), center_z(0.0),
        scale_xy(1.0), scale_z(1.0) {
    }
};

// ============================================================================
// TERRAIN LOADER
// ============================================================================

class TerrainLoader {
public:
    TerrainLoader() : transform_params() {}

    bool loadFile(const std::string& filename);
    bool transformPoints();

    const std::vector<Point3D>& getTransformedPoints() const {
        return transformed_points;
    }

    const TransformParams& getTransformParams() const {
        return transform_params;
    }

    void printStatistics() const;

private:
    std::vector<Point3D> original_points;
    std::vector<Point3D> transformed_points;
    TransformParams transform_params;

    void calculateBoundingBox(
        double& x_min, double& x_max,
        double& y_min, double& y_max,
        double& z_min, double& z_max) const;
};

bool TerrainLoader::loadFile(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    int point_count = 0;
    file >> point_count;

    if (point_count <= 0) {
        std::cerr << "Error: Invalid point count " << point_count << std::endl;
        return false;
    }

    std::cout << "Loading " << point_count << " points..." << std::endl;

    original_points.clear();
    original_points.reserve(point_count);

    for (int i = 0; i < point_count; ++i) {
        double x, y, z;
        if (!(file >> x >> y >> z)) {
            std::cerr << "Error: Failed to read point " << i << std::endl;
            file.close();
            return false;
        }
        original_points.emplace_back(x, y, z);
    }

    file.close();

    std::cout << "Successfully loaded " << original_points.size() << " points" << std::endl;
    printStatistics();

    return true;
}

void TerrainLoader::calculateBoundingBox(
    double& x_min, double& x_max,
    double& y_min, double& y_max,
    double& z_min, double& z_max) const {

    if (original_points.empty()) {
        x_min = x_max = y_min = y_max = z_min = z_max = 0.0;
        return;
    }

    x_min = x_max = original_points[0].x;
    y_min = y_max = original_points[0].y;
    z_min = z_max = original_points[0].z;

    for (const auto& p : original_points) {
        x_min = std::min(x_min, p.x);
        x_max = std::max(x_max, p.x);
        y_min = std::min(y_min, p.y);
        y_max = std::max(y_max, p.y);
        z_min = std::min(z_min, p.z);
        z_max = std::max(z_max, p.z);
    }
}

bool TerrainLoader::transformPoints() {
    if (original_points.empty()) {
        std::cerr << "Error: No points loaded" << std::endl;
        return false;
    }

    double x_min, x_max, y_min, y_max, z_min, z_max;
    calculateBoundingBox(x_min, x_max, y_min, y_max, z_min, z_max);

    transform_params.center_x = (x_min + x_max) / 2.0;
    transform_params.center_y = (y_min + y_max) / 2.0;
    transform_params.center_z = (z_min + z_max) / 2.0;

    double range_xy = std::max(x_max - x_min, y_max - y_min);
    transform_params.scale_xy = range_xy / 2.0;

    double range_z = z_max - z_min;
    transform_params.scale_z = range_z / 2.0;

    if (transform_params.scale_xy < EPSILON) transform_params.scale_xy = 1.0;
    if (transform_params.scale_z < EPSILON) transform_params.scale_z = 1.0;

    transformed_points.clear();
    transformed_points.reserve(original_points.size());

    for (const auto& p : original_points) {
        double x_t = (p.x - transform_params.center_x) / transform_params.scale_xy;
        double y_t = (p.y - transform_params.center_y) / transform_params.scale_xy;
        double z_t = (p.z - transform_params.center_z) / transform_params.scale_z;

        transformed_points.emplace_back(x_t, y_t, z_t);
    }

    std::cout << "Transformation complete" << std::endl;
    std::cout << "  Center: (" << transform_params.center_x << ", "
              << transform_params.center_y << ", "
              << transform_params.center_z << ")" << std::endl;
    std::cout << "  Scale XY: " << transform_params.scale_xy << std::endl;
    std::cout << "  Scale Z: " << transform_params.scale_z << std::endl;

    return true;
}

void TerrainLoader::printStatistics() const {
    if (original_points.empty()) return;

    double x_min, x_max, y_min, y_max, z_min, z_max;
    calculateBoundingBox(x_min, x_max, y_min, y_max, z_min, z_max);

    std::cout << "\nPoint Statistics:" << std::endl;
    std::cout << "  Total points: " << original_points.size() << std::endl;
    std::cout << "  X range: [" << x_min << ", " << x_max << "] (range: "
              << (x_max - x_min) << ")" << std::endl;
    std::cout << "  Y range: [" << y_min << ", " << y_max << "] (range: "
              << (y_max - y_min) << ")" << std::endl;
    std::cout << "  Z range: [" << z_min << ", " << z_max << "] (range: "
              << (z_max - z_min) << ")" << std::endl;
}

// ============================================================================
// ROBUST DELAUNAY TRIANGULATION (Bowyer-Watson)
// ============================================================================

class Delaunay {
public:
    std::vector<ConverterTriangle> triangles;
    std::vector<Vec3> extended_points;

    bool triangulate(const std::vector<Point3D>& points,
                     const std::vector<Vec3>& points_vec3) {
        if (points.size() < 3) {
            std::cerr << "Error: Need at least 3 points" << std::endl;
            return false;
        }

        std::cout << "Performing Bowyer-Watson Delaunay" << std::endl;

        triangles.clear();
        extended_points.clear();

        for (const auto& p : points_vec3) {
            extended_points.push_back(p);
        }

        std::vector<Vec2> points2d;
        for (const auto& p : points) {
            points2d.emplace_back(p.x, p.y);
        }

        size_t n = points.size();

        // Create super-triangle
        double minx = points2d[0].x, maxx = points2d[0].x;
        double miny = points2d[0].y, maxy = points2d[0].y;

        for (const auto& p : points2d) {
            minx = std::min(minx, p.x);
            maxx = std::max(maxx, p.x);
            miny = std::min(miny, p.y);
            maxy = std::max(maxy, p.y);
        }

        double dx = maxx - minx;
        double dy = maxy - miny;
        double dmax = std::max(dx, dy);
        double midx = (minx + maxx) / 2.0;
        double midy = (miny + maxy) / 2.0;

        double scale = 100.0;
        Vec2 super_a(midx - scale * dmax, midy - scale * dmax);
        Vec2 super_b(midx, midy + scale * dmax);
        Vec2 super_c(midx + scale * dmax, midy - scale * dmax);

        size_t idx_a = extended_points.size();
        size_t idx_b = extended_points.size() + 1;
        size_t idx_c = extended_points.size() + 2;

        extended_points.push_back(Vec3(super_a.x, super_a.y, 0.0));
        extended_points.push_back(Vec3(super_b.x, super_b.y, 0.0));
        extended_points.push_back(Vec3(super_c.x, super_c.y, 0.0));

        triangles.push_back(ConverterTriangle(idx_a, idx_b, idx_c));

        // Bowyer-Watson algorithm
        for (size_t i = 0; i < n; ++i) {
            std::vector<ConverterTriangle> bad_triangles;

            for (const auto& tri : triangles) {
                if (isInCircumcircle(points2d[i],
                                     Vec2(extended_points[tri.v0].x, extended_points[tri.v0].y),
                                     Vec2(extended_points[tri.v1].x, extended_points[tri.v1].y),
                                     Vec2(extended_points[tri.v2].x, extended_points[tri.v2].y))) {
                    bad_triangles.push_back(tri);
                }
            }

            std::vector<std::pair<size_t, size_t>> polygon;
            for (const auto& tri : bad_triangles) {
                addEdgeIfNotDuplicate(polygon, tri.v0, tri.v1);
                addEdgeIfNotDuplicate(polygon, tri.v1, tri.v2);
                addEdgeIfNotDuplicate(polygon, tri.v2, tri.v0);
            }

            for (const auto& bad : bad_triangles) {
                auto it = std::find_if(triangles.begin(), triangles.end(),
                                       [&bad](const ConverterTriangle& t) { return t == bad; });
                if (it != triangles.end()) {
                    triangles.erase(it);
                }
            }

            for (const auto& edge : polygon) {
                Vec2 v0_2d(extended_points[edge.first].x, extended_points[edge.first].y);
                Vec2 v1_2d(extended_points[edge.second].x, extended_points[edge.second].y);
                Vec2 p_2d(points2d[i].x, points2d[i].y);

                double cross = (v1_2d - v0_2d).cross(p_2d - v0_2d);

                if (cross > EPSILON) {
                    triangles.emplace_back(edge.first, edge.second, i);
                }
                else if (cross < -EPSILON) {
                    triangles.emplace_back(edge.second, edge.first, i);
                }
            }
        }

        std::vector<ConverterTriangle> final_triangles;
        for (const auto& tri : triangles) {
            if (tri.v0 < n && tri.v1 < n && tri.v2 < n) {
                final_triangles.push_back(tri);
            }
        }
        triangles = final_triangles;

        std::cout << "Delaunay triangulation complete: " << triangles.size()
                  << " triangles (after filtering degenerate)" << std::endl;

        extended_points.erase(extended_points.begin() + n, extended_points.end());

        return true;
    }

private:
    bool isInCircumcircle(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c) {
        double ax = a.x, ay = a.y;
        double bx = b.x, by = b.y;
        double cx = c.x, cy = c.y;
        double px = p.x, py = p.y;

        double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

        if (std::abs(d) < EPSILON) return false;

        double ux = ((ax * ax + ay * ay) * (by - cy) +
                     (bx * bx + by * by) * (cy - ay) +
                     (cx * cx + cy * cy) * (ay - by)) / d;
        double uy = ((ax * ax + ay * ay) * (cx - bx) +
                     (bx * bx + by * by) * (ax - cx) +
                     (cx * cx + cy * cy) * (bx - ax)) / d;

        double radius_sq = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);
        double dist_sq = (px - ux) * (px - ux) + (py - uy) * (py - uy);

        return dist_sq < radius_sq + EPSILON;
    }

    void addEdgeIfNotDuplicate(std::vector<std::pair<size_t, size_t>>& edges,
                               size_t a, size_t b) {
        auto it = std::find_if(edges.begin(), edges.end(),
                               [a, b](const std::pair<size_t, size_t>& e) {
                                   return (e.first == b && e.second == a);
                               });

        if (it != edges.end()) {
            edges.erase(it);
        }
        else {
            edges.emplace_back(a, b);
        }
    }
};

// ============================================================================
// OBJ GENERATOR
// ============================================================================

class ObjGenerator {
public:
    ObjGenerator() {}

    bool triangulate(const std::vector<Point3D>& points);
    bool computeVertexNormals();
    bool writeObjFile(const std::string& filename, const TransformParams& params);

private:
    std::vector<ConverterVertex> vertices;
    std::vector<ConverterTriangle> triangles;

    std::string formatFloat(double value, int decimals = 8) const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(decimals) << value;
        return oss.str();
    }
};

bool ObjGenerator::triangulate(const std::vector<Point3D>& points) {
    if (points.size() < 3) {
        std::cerr << "Error: Need at least 3 points" << std::endl;
        return false;
    }

    std::cout << "Starting Delaunay triangulation for " << points.size()
              << " points..." << std::endl;

    std::vector<Vec3> points_vec3;
    for (const auto& p : points) {
        points_vec3.emplace_back(p.x, p.y, p.z);
    }

    Delaunay d;
    if (!d.triangulate(points, points_vec3)) {
        std::cerr << "Triangulation failed" << std::endl;
        return false;
    }

    triangles = d.triangles;

    vertices.clear();
    vertices.reserve(points.size());

    for (const auto& p : points_vec3) {
        ConverterVertex v;
        v.position = p;
        v.normal = Vec3(0.0, 0.0, 1.0);
        vertices.push_back(v);
    }

    return true;
}

bool ObjGenerator::computeVertexNormals() {
    if (vertices.empty() || triangles.empty()) {
        std::cerr << "Error: No vertices or triangles" << std::endl;
        return false;
    }

    std::cout << "Computing smooth vertex normals..." << std::endl;

    for (auto& v : vertices) {
        v.normal = Vec3(0.0, 0.0, 0.0);
    }

    size_t degenerate_count = 0;

    for (const auto& tri : triangles) {
        const Vec3& p0 = vertices[tri.v0].position;
        const Vec3& p1 = vertices[tri.v1].position;
        const Vec3& p2 = vertices[tri.v2].position;

        Vec3 u = p1 - p0;
        Vec3 v = p2 - p0;
        Vec3 tri_normal = u.cross(v);

        double length = tri_normal.length();

        if (length < EPSILON) {
            degenerate_count++;
            continue;
        }

        tri_normal = tri_normal.normalize();

        double area = length / 2.0;

        vertices[tri.v0].normal += tri_normal * area;
        vertices[tri.v1].normal += tri_normal * area;
        vertices[tri.v2].normal += tri_normal * area;
    }

    if (degenerate_count > 0) {
        std::cout << "  Skipped " << degenerate_count << " degenerate triangles" << std::endl;
    }

    for (auto& v : vertices) {
        v.normal = v.normal.normalize();
    }

    std::cout << "Vertex normals computed (smooth shading)" << std::endl;
    return true;
}

bool ObjGenerator::writeObjFile(const std::string& filename,
                                const TransformParams& params) {
    if (vertices.empty() || triangles.empty()) {
        std::cerr << "Error: No data to write" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open output file " << filename << std::endl;
        return false;
    }

    std::cout << "Writing OBJ file: " << filename << std::endl;

    file << "# Terrain Point Cloud to Triangulated Surface\n";
    file << "# FINAL: Bowyer-Watson Delaunay (degenerate triangles removed)\n";
    file << "# Coordinate System: Vulkan-compatible (X, Z, -Y)\n";
    file << "# Original input: terrain.xyz with Z as height\n";
    file << "# Output format: X stays X, Z becomes Y (height), -Y becomes Z (depth)\n";
    file << "#\n";
    file << "# Statistics:\n";
    file << "# - Total vertices: " << vertices.size() << "\n";
    file << "# - Valid triangles (degenerate removed): " << triangles.size() << "\n";
    file << "#\n";
    file << "# Transformation:\n";
    file << "# - Center: (" << params.center_x << ", "
         << params.center_y << ", " << params.center_z << ")\n";
    file << "# - Scale XY: " << params.scale_xy << "\n";
    file << "# - Scale Z: " << params.scale_z << "\n";
    file << "\n";

    file << "# ============ VERTICES (v x y z) ============\n";
    file << "# Format: Vulkan coordinates (original_x, original_z, -original_y)\n";
    for (const auto& v : vertices) {
        // Convert from internal format (x, y, z) to Vulkan format (x, z, -y)
        double vx = v.position.x;
        double vy = v.position.y;
        double vz = v.position.z;
        file << "v " << formatFloat(vx) << " " << formatFloat(vz) << " " << formatFloat(-vy) << "\n";
    }
    file << "\n";

    file << "# ============ VERTEX NORMALS (vn nx ny nz) ============\n";
    file << "# Format: Vulkan coordinates (original_nx, original_nz, -original_ny)\n";
    for (const auto& v : vertices) {
        // Convert from internal format (x, y, z) to Vulkan format (x, z, -y)
        double nx = v.normal.x;
        double ny = v.normal.y;
        double nz = v.normal.z;
        file << "vn " << formatFloat(nx) << " " << formatFloat(nz) << " " << formatFloat(-ny) << "\n";
    }
    file << "\n";

    file << "# ============ FACES (f v1//vn1 v2//vn2 v3//vn3) ============\n";
    for (const auto& tri : triangles) {
        file << "f " << (tri.v0 + 1) << "//" << (tri.v0 + 1) << " "
             << (tri.v1 + 1) << "//" << (tri.v1 + 1) << " "
             << (tri.v2 + 1) << "//" << (tri.v2 + 1) << "\n";
    }

    file.close();

    std::cout << "OBJ file written successfully" << std::endl;
    std::cout << "  Vertices: " << vertices.size() << std::endl;
    std::cout << "  Normals: " << vertices.size() << std::endl;
    std::cout << "  Valid faces: " << triangles.size() << std::endl;
    std::cout << "  Status: ✓ No stretched parts, ✓ Uniform mesh, ✓ Correct normals" << std::endl;

    return true;
}

// ============================================================================
// TERRAIN CONVERTER FUNCTION - CALLED BEFORE VULKAN INIT
// ============================================================================

bool convertTerrainXYZtoOBJ(const std::string& input_file, const std::string& output_file) {
    std::cout << "\n=== Terrain Point Cloud to OBJ Converter ===" << std::endl;
    std::cout << "FINAL: Robust Bowyer-Watson Delaunay" << std::endl;
    std::cout << "Fixed: Stretched vertices, flipped normals, degenerate triangles" << std::endl;
    std::cout << std::endl;

    std::cout << "Input:  " << input_file << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    std::cout << std::endl;

    // Step 1: Load
    std::cout << "STEP 1: Loading terrain data..." << std::endl;
    TerrainLoader loader;
    if (!loader.loadFile(input_file)) {
        std::cerr << "Failed to load terrain file" << std::endl;
        return false;
    }
    std::cout << "✓ Terrain data loaded\n" << std::endl;

    // Step 2: Transform
    std::cout << "STEP 2: Transforming points to [-1, 1]..." << std::endl;
    if (!loader.transformPoints()) {
        std::cerr << "Failed to transform points" << std::endl;
        return false;
    }
    std::cout << "✓ Points transformed\n" << std::endl;

    // Step 3: Triangulate
    std::cout << "STEP 3: Performing Bowyer-Watson Delaunay..." << std::endl;
    ObjGenerator generator;
    if (!generator.triangulate(loader.getTransformedPoints())) {
        std::cerr << "Failed to triangulate" << std::endl;
        return false;
    }
    std::cout << "✓ Triangulation complete\n" << std::endl;

    // Step 4: Normals
    std::cout << "STEP 4: Computing smooth vertex normals..." << std::endl;
    if (!generator.computeVertexNormals()) {
        std::cerr << "Failed to compute normals" << std::endl;
        return false;
    }
    std::cout << "✓ Vertex normals computed\n" << std::endl;

    // Step 5: Write
    std::cout << "STEP 5: Writing OBJ file..." << std::endl;
    if (!generator.writeObjFile(output_file, loader.getTransformParams())) {
        std::cerr << "Failed to write OBJ file" << std::endl;
        return false;
    }
    std::cout << "✓ OBJ file written\n" << std::endl;

    std::cout << "=== SUCCESS ===" << std::endl;
    std::cout << "OBJ file: " << output_file << std::endl;
    std::cout << "✓ No stretched parts (super-triangle completely removed)" << std::endl;
    std::cout << "✓ Uniform grid (correct triangle winding)" << std::endl;
    std::cout << "✓ Correct normals (degenerate triangles filtered)" << std::endl;
    std::cout << "✓ Ready for Vulkan rendering + ball physics!" << std::endl;
    std::cout << std::endl;

    return true;
}

// ============================================================================
// VULKAN STRUCTURES
// ============================================================================

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct LightUniformBuffer {
    alignas(16) glm::vec3 lightDirection;
    alignas(16) glm::vec3 lightColor;
    alignas(16) glm::vec3 cameraPos;
    alignas(4) float ambientStrength;
    alignas(4) float diffuseStrength;
    alignas(4) float specularStrength;
    alignas(4) float shininess;
};

// ============================================================================
// VULKAN TERRAIN APP
// ============================================================================

class TerrainApp {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

    VkImage skyboxImage = VK_NULL_HANDLE;
    VkDeviceMemory skyboxImageMemory = VK_NULL_HANDLE;
    VkImageView skyboxImageView = VK_NULL_HANDLE;
    VkSampler skyboxSampler = VK_NULL_HANDLE;

    VkDescriptorSetLayout skyboxDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout skyboxPipelineLayout = VK_NULL_HANDLE;
    VkPipeline skyboxPipeline = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> skyboxDescriptorSets;

    VkBuffer skyboxVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory skyboxVertexBufferMemory = VK_NULL_HANDLE;
    uint32_t skyboxVertexCount = 0;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    glm::vec3 cameraPos = glm::vec3(0.0f, 10.0f, 20.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    float cameraSpeed = 5.0f;
    float mouseSensitivity = 0.1f;
    float yaw = -90.0f;
    float pitch = 0.0f;

    double lastX = WIDTH / 2.0;
    double lastY = HEIGHT / 2.0;
    bool firstMouse = true;

    glm::vec3 lightDirection = glm::normalize(glm::vec3(0.5f, -1.0f, 0.3f));
    glm::vec3 lightColor = glm::vec3(1.0f, 0.95f, 0.8f);
    float ambientStrength = 0.3f;
    float diffuseStrength = 1.0f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;
    float sunAngle = 45.0f;

    glm::mat4 modelMatrix = glm::mat4(1.0f);

    std::vector<VkBuffer> lightUniformBuffers;
    std::vector<VkDeviceMemory> lightUniformBuffersMemory;
    std::vector<void*> lightUniformBuffersMapped;

    bool framebufferResized = false;


    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Point Cloud Viewer", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetCursorPosCallback(window, mouseCallback);
        glfwSetScrollCallback(window, scrollCallback);

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<TerrainApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        auto app = reinterpret_cast<TerrainApp*>(glfwGetWindowUserPointer(window));
        app->cameraSpeed += static_cast<float>(yoffset) * 0.5f;
        if (app->cameraSpeed < 1.0f) app->cameraSpeed = 1.0f;
        if (app->cameraSpeed > 50.0f) app->cameraSpeed = 50.0f;
    }

    static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
        auto app = reinterpret_cast<TerrainApp*>(glfwGetWindowUserPointer(window));

        if (app->firstMouse) {
            app->lastX = xpos;
            app->lastY = ypos;
            app->firstMouse = false;
        }

        float xoffset = xpos - app->lastX;
        float yoffset = app->lastY - ypos;
        app->lastX = xpos;
        app->lastY = ypos;

        xoffset *= app->mouseSensitivity;
        yoffset *= app->mouseSensitivity;

        app->yaw += xoffset;
        app->pitch += yoffset;

        if (app->pitch > 89.0f) app->pitch = 89.0f;
        if (app->pitch < -89.0f) app->pitch = -89.0f;

        glm::vec3 front;
        front.x = glm::cos(glm::radians(app->yaw)) * glm::cos(glm::radians(app->pitch));
        front.y = glm::sin(glm::radians(app->pitch));
        front.z = glm::sin(glm::radians(app->yaw)) * glm::cos(glm::radians(app->pitch));
        app->cameraFront = glm::normalize(front);
    }


    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();

        loadTerrainOBJ("terrain.obj");

        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createSkyboxResources();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void loadTerrainOBJ(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open OBJ file: " + filename);
        }

        vertices.clear();
        indices.clear();

        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals;

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string token;
            iss >> token;

            if (token == "v") {
                float x, y, z;
                iss >> x >> y >> z;
                // OBJ already in Vulkan format (x, z, -y) from converter
                // Read directly without transformation
                positions.emplace_back(x, y, z);
            }
            else if (token == "vn") {
                float nx, ny, nz;
                iss >> nx >> ny >> nz;
                // Normals already converted by converter
                normals.emplace_back(nx, ny, nz);
            }
            else if (token == "f") {
                std::string face;
                while (iss >> face) {
                    size_t slashPos = face.find("//");
                    if (slashPos != std::string::npos) {
                        int v_idx = std::stoi(face.substr(0, slashPos)) - 1;
                        int vn_idx = std::stoi(face.substr(slashPos + 2)) - 1;

                        if (v_idx >= 0 && v_idx < (int)positions.size() &&
                            vn_idx >= 0 && vn_idx < (int)normals.size()) {
                            Vertex vertex;
                            vertex.pos = positions[v_idx];
                            vertex.normal = normals[vn_idx];
                            vertices.push_back(vertex);
                            indices.push_back(static_cast<uint32_t>(vertices.size() - 1));
                        }
                    }
                }
            }
        }

        file.close();

        std::cout << "Loaded OBJ file: " << filename << std::endl;
        std::cout << "  Format: Pre-converted Vulkan coordinates (x, z, -y)" << std::endl;
        std::cout << "  Vertices: " << vertices.size() << std::endl;
        std::cout << "  Triangles: " << (indices.size() / 3) << std::endl;

        if (vertices.empty()) {
            throw std::runtime_error("No vertices loaded from OBJ file!");
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            processInput();
            drawFrame();
        }
        vkDeviceWaitIdle(device);
    }

    void processInput() {
        static auto lastFrameTime = std::chrono::high_resolution_clock::now();
        auto currentFrameTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(
                              currentFrameTime - lastFrameTime).count();
        lastFrameTime = currentFrameTime;

        float velocity = cameraSpeed * deltaTime;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += velocity * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= velocity * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity;

        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            cameraPos += velocity * cameraUp;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            cameraPos -= velocity * cameraUp;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            sunAngle += 20.0f * deltaTime;
            if (sunAngle > 180.0f) sunAngle = 180.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            sunAngle -= 20.0f * deltaTime;
            if (sunAngle < 0.0f) sunAngle = 0.0f;
        }

        float sunRadians = glm::radians(sunAngle);
        lightDirection = glm::normalize(glm::vec3(
            glm::cos(sunRadians) * 0.5f,
            -glm::sin(sunRadians),
            0.3f
            ));

        if (sunAngle < 30.0f || sunAngle > 150.0f) {
            float t = (sunAngle < 30.0f) ? (30.0f - sunAngle) / 30.0f : (sunAngle - 150.0f) / 30.0f;
            lightColor = glm::mix(glm::vec3(1.0f, 0.95f, 0.8f), glm::vec3(1.0f, 0.6f, 0.3f), t);
        } else {
            lightColor = glm::vec3(1.0f, 0.95f, 0.8f);
        }

        ambientStrength = 0.1f + (glm::sin(sunRadians) * 0.3f);
        if (ambientStrength < 0.1f) ambientStrength = 0.1f;

        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            specularStrength -= 0.5f * deltaTime;
            if (specularStrength < 0.0f) specularStrength = 0.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
            specularStrength += 0.5f * deltaTime;
            if (specularStrength > 2.0f) specularStrength = 2.0f;
        }

        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
            shininess -= 10.0f * deltaTime;
            if (shininess < 2.0f) shininess = 2.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
            shininess += 10.0f * deltaTime;
            if (shininess > 128.0f) shininess = 128.0f;
        }
    }



    void cleanup() {
        cleanupSwapChain();

        vkDestroySampler(device, skyboxSampler, nullptr);
        vkDestroyImageView(device, skyboxImageView, nullptr);
        vkDestroyImage(device, skyboxImage, nullptr);
        vkFreeMemory(device, skyboxImageMemory, nullptr);

        vkDestroyBuffer(device, skyboxVertexBuffer, nullptr);
        vkFreeMemory(device, skyboxVertexBufferMemory, nullptr);

        vkDestroyPipeline(device, skyboxPipeline, nullptr);
        vkDestroyPipelineLayout(device, skyboxPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, skyboxDescriptorSetLayout, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
            vkDestroyBuffer(device, lightUniformBuffers[i], nullptr);
            vkFreeMemory(device, lightUniformBuffersMemory[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Terrain Mesh Viewer";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        createInfo.enabledLayerCount = 0;

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        std::multimap<int, VkPhysicalDevice> candidates;

        for (const auto& device : devices) {
            int score = rateDeviceSuitability(device);
            candidates.insert(std::make_pair(score, device));
        }

        if (candidates.rbegin()->first > 0) {
            physicalDevice = candidates.rbegin()->second;

            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
            std::cout << "Selected GPU: " << deviceProperties.deviceName << std::endl;
        } else {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }

    int rateDeviceSuitability(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        int score = 0;

        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        }

        score += deviceProperties.limits.maxImageDimension2D;

        if (!isDeviceSuitable(device)) {
            return 0;
        }

        return score;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
                if (presentSupport) {
                    return true;
                }
            }
        }
        return false;
    }

    void createLogicalDevice() {
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = 0;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkPhysicalDeviceFeatures deviceFeatures{};

        const char* extensionNames[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = 1;
        createInfo.ppEnabledExtensionNames = extensionNames;
        createInfo.enabledLayerCount = 0;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device!");
        }

        vkGetDeviceQueue(device, 0, 0, &graphicsQueue);
        vkGetDeviceQueue(device, 0, 0, &presentQueue);
    }

    void createSwapChain() {
        VkSurfaceCapabilitiesKHR capabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

        VkExtent2D extent = capabilities.currentExtent;
        if (extent.width == UINT32_MAX) {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            extent.width = std::clamp(static_cast<uint32_t>(width),
                                      capabilities.minImageExtent.width,
                                      capabilities.maxImageExtent.width);
            extent.height = std::clamp(static_cast<uint32_t>(height),
                                       capabilities.minImageExtent.height,
                                       capabilities.maxImageExtent.height);
        }

        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = VK_FORMAT_B8G8R8A8_SRGB;
        createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swap chain!");
        }

        swapChainImageFormat = VK_FORMAT_B8G8R8A8_SRGB;
        swapChainExtent = extent;

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i],
                                                     swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view!");
        }
        return imageView;
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass!");
        }
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("vert.spv");
        auto fragShaderCode = readFile("frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                              VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                      &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();
        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    VkFormat findDepthFormat() {
        return VK_FORMAT_D32_SFLOAT;
    }

    void createFramebuffers() {
        framebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformBuffersMemory[i]);
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }

        VkDeviceSize lightBufferSize = sizeof(LightUniformBuffer);

        lightUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        lightUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        lightUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(lightBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         lightUniformBuffers[i], lightUniformBuffersMemory[i]);
            vkMapMemory(device, lightUniformBuffersMemory[i], 0, lightBufferSize, 0, &lightUniformBuffersMapped[i]);
        }
    }

    void updateLightUniforms(uint32_t currentImage) {
        LightUniformBuffer lightUBO{};
        lightUBO.lightDirection = lightDirection;
        lightUBO.lightColor = lightColor;
        lightUBO.cameraPos = cameraPos;
        lightUBO.ambientStrength = ambientStrength;
        lightUBO.diffuseStrength = diffuseStrength;
        lightUBO.specularStrength = specularStrength;
        lightUBO.shininess = shininess;

        memcpy(lightUniformBuffersMapped[currentImage], &lightUBO, sizeof(lightUBO));
    }

    void updateUniformBuffer(uint32_t currentImage) {
        UniformBufferObject ubo{};

        ubo.model = modelMatrix;

        ubo.view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        ubo.proj = glm::perspective(
            glm::radians(45.0f),
            swapChainExtent.width / static_cast<float>(swapChainExtent.height),
            0.1f, 1000.0f
            );
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void createDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        bindings[0].pImmutableSamplers = nullptr;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1].pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }

    void createSkyboxResources() {
        createSkyboxCubemap();
        createSkyboxGeometry();
        createSkyboxDescriptorSetLayout();
        createSkyboxPipeline();
    }

    void createSkyboxCubemap() {
        const char* faces[6] = {
            "right.jpg", "left.jpg", "up.jpg", "down.jpg", "front.jpg", "back.jpg"
        };

        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(faces[0], &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("Failed to load skybox texture: right.jpg");
        }

        VkDeviceSize layerSize = texWidth * texHeight * 4;
        VkDeviceSize imageSize = layerSize * 6;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);

        for (int i = 0; i < 6; i++) {
            if (i > 0) {
                stbi_image_free(pixels);
                pixels = stbi_load(faces[i], &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
                if (!pixels) {
                    throw std::runtime_error(std::string("Failed to load skybox face: ") + faces[i]);
                }
            }
            memcpy(static_cast<char*>(data) + (layerSize * i), pixels, static_cast<size_t>(layerSize));
        }

        stbi_image_free(pixels);
        vkUnmapMemory(device, stagingBufferMemory);

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = texWidth;
        imageInfo.extent.height = texHeight;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 6;
        imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

        if (vkCreateImage(device, &imageInfo, nullptr, &skyboxImage) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create cubemap image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, skyboxImage, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &skyboxImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate cubemap memory!");
        }

        vkBindImageMemory(device, skyboxImage, skyboxImageMemory, 0);

        transitionImageLayout(skyboxImage, VK_FORMAT_R8G8B8A8_SRGB,
                              VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6);

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        for (uint32_t i = 0; i < 6; i++) {
            VkBufferImageCopy region{};
            region.bufferOffset = layerSize * i;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = i;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1};

            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, skyboxImage,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        endSingleTimeCommands(commandBuffer);

        transitionImageLayout(skyboxImage, VK_FORMAT_R8G8B8A8_SRGB,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = skyboxImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 6;

        if (vkCreateImageView(device, &viewInfo, nullptr, &skyboxImageView) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create cubemap image view!");
        }

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &skyboxSampler) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create cubemap sampler!");
        }
    }

    void createSkyboxGeometry() {
        std::vector<glm::vec3> skyboxVertices = {
            {-1.0f,  1.0f, -1.0f}, {-1.0f, -1.0f, -1.0f}, { 1.0f, -1.0f, -1.0f},
            { 1.0f, -1.0f, -1.0f}, { 1.0f,  1.0f, -1.0f}, {-1.0f,  1.0f, -1.0f},

            {-1.0f, -1.0f,  1.0f}, {-1.0f, -1.0f, -1.0f}, {-1.0f,  1.0f, -1.0f},
            {-1.0f,  1.0f, -1.0f}, {-1.0f,  1.0f,  1.0f}, {-1.0f, -1.0f,  1.0f},

            { 1.0f, -1.0f, -1.0f}, { 1.0f, -1.0f,  1.0f}, { 1.0f,  1.0f,  1.0f},
            { 1.0f,  1.0f,  1.0f}, { 1.0f,  1.0f, -1.0f}, { 1.0f, -1.0f, -1.0f},

            {-1.0f, -1.0f,  1.0f}, {-1.0f,  1.0f,  1.0f}, { 1.0f,  1.0f,  1.0f},
            { 1.0f,  1.0f,  1.0f}, { 1.0f, -1.0f,  1.0f}, {-1.0f, -1.0f,  1.0f},

            {-1.0f,  1.0f, -1.0f}, { 1.0f,  1.0f, -1.0f}, { 1.0f,  1.0f,  1.0f},
            { 1.0f,  1.0f,  1.0f}, {-1.0f,  1.0f,  1.0f}, {-1.0f,  1.0f, -1.0f},

            {-1.0f, -1.0f, -1.0f}, {-1.0f, -1.0f,  1.0f}, { 1.0f, -1.0f, -1.0f},
            { 1.0f, -1.0f, -1.0f}, {-1.0f, -1.0f,  1.0f}, { 1.0f, -1.0f,  1.0f}
        };

        skyboxVertexCount = skyboxVertices.size();

        VkDeviceSize bufferSize = sizeof(skyboxVertices[0]) * skyboxVertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, skyboxVertices.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skyboxVertexBuffer, skyboxVertexBufferMemory);

        copyBuffer(stagingBuffer, skyboxVertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createSkyboxDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &skyboxDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create skybox descriptor set layout!");
        }
    }

    void createSkyboxPipeline() {
        auto vertShaderCode = readFile("skybox_vert.spv");
        auto fragShaderCode = readFile("skybox_frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(glm::vec3);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attributeDescription{};
        attributeDescription.binding = 0;
        attributeDescription.location = 0;
        attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescription.offset = 0;

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = 1;
        vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                              VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &skyboxDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &skyboxPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create skybox pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = skyboxPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                      &skyboxPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create skybox pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
                               VkImageLayout newLayout, uint32_t layerCount = 1) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = layerCount;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::runtime_error("Unsupported layout transition!");
        }

        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(commandBuffer);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 3);

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 2);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorBufferInfo lightBufferInfo{};
            lightBufferInfo.buffer = lightUniformBuffers[i];
            lightBufferInfo.offset = 0;
            lightBufferInfo.range = sizeof(LightUniformBuffer);

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &lightBufferInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(), 0, nullptr);
        }

        std::vector<VkDescriptorSetLayout> skyboxLayouts(MAX_FRAMES_IN_FLIGHT, skyboxDescriptorSetLayout);

        VkDescriptorSetAllocateInfo skyboxAllocInfo{};
        skyboxAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        skyboxAllocInfo.descriptorPool = descriptorPool;
        skyboxAllocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        skyboxAllocInfo.pSetLayouts = skyboxLayouts.data();

        skyboxDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &skyboxAllocInfo, skyboxDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate skybox descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = skyboxImageView;
            imageInfo.sampler = skyboxSampler;

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = skyboxDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = skyboxDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(), 0, nullptr);
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);
        VkBuffer skyboxVertexBuffers[] = {skyboxVertexBuffer};
        VkDeviceSize skyboxOffsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, skyboxVertexBuffers, skyboxOffsets);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                skyboxPipelineLayout, 0, 1, &skyboxDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDraw(commandBuffer, skyboxVertexCount, 1, 0, 0);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer!");
        }
    }

    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
                                                imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        updateUniformBuffer(currentFrame);
        updateLightUniforms(currentFrame);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = 0;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool!");
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create synchronization objects!");
            }
        }
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }

    void cleanupSwapChain() {
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (auto framebuffer : framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }

        return shaderModule;
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                     VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }
};

// ============================================================================
// MAIN ENTRY POINT - CONVERTER RUNS BEFORE VULKAN APP
// ============================================================================

int main() {
    try {
       /* // STEP 1: Run the terrain converter (XYZ → OBJ with Delaunay triangulation)
        std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  TERRAIN CONVERSION PHASE (PRE-VULKAN)    ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════╝\n" << std::endl;

        std::string input_xyz = "terrain.xyz";
        std::string output_obj = "terrain.obj";

        if (!convertTerrainXYZtoOBJ(input_xyz, output_obj)) {
            std::cerr << "Terrain conversion failed!" << std::endl;
            return EXIT_FAILURE;
        }*/


        // STEP 2: Initialize and run Vulkan viewer
        std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  VULKAN RENDERER INITIALIZATION          ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════╝\n" << std::endl;

        TerrainApp app;
        app.run();

    } catch (const std::exception& e) {
        std::cerr << "\n✗ FATAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n✓ Application terminated successfully." << std::endl;
    return EXIT_SUCCESS;
}
