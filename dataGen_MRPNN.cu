#include "volume.hpp"
#include "renderResources.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>
#include <filesystem>

namespace {

constexpr int kStencilPointCount = 192;
constexpr int kOutputColumns = kStencilPointCount * 3 + 4;
constexpr int kSamplesPerModel = 250000;
constexpr int kConditionCountPerModel = 50;
constexpr int kSamplesPerCondition = kSamplesPerModel / kConditionCountPerModel;
static_assert(kSamplesPerModel % kConditionCountPerModel == 0, "Paper sampling requires an integer number of samples per condition.");
constexpr int kPathTraceScatterCount = 512;
constexpr int kPathTraceSampleCount = 1024;
constexpr float kMaxG = 0.857f;
constexpr float kEpsilonMin = 0.5f;
constexpr float kEpsilonMax = 8.0f;
constexpr float kMinZeta = 0.1f;
constexpr float kMaxZeta = 1.0f;
constexpr float kAlbedoExponent = 4.0f;

float hash1() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

struct MRPNNSamplePoint {
    float3 position;
    float3 viewDir;
    float3 lightDir;
    float epsilon;
    float g;
    float albedo;
};

struct MRPNNDescriptorRow {
    std::array<float, kStencilPointCount> density{};
    std::array<float, kStencilPointCount> transmittance{};
    std::array<float, kStencilPointCount> phase{};
    float g = 0.0f;
    float zetaPowAlpha = 1.0f;
    float gamma = 0.0f;
    float targetPredictRadiance = 0.0f;
};

struct BasisPair {
    float3x3 main;
    float3x3 light;
};

struct DeviceDescriptorRow {
    float density[kStencilPointCount];
    float transmittance[kStencilPointCount];
    float phase[kStencilPointCount];
    float gamma;
};

constexpr float3 kSpn0[8] = {
    float3{0.000000f, 0.000000f, 0.000000f},
    float3{1.000000f, 0.000000f, 0.000000f},
    float3{0.666667f, -0.549602f, 0.503481f},
    float3{0.333333f, 0.082426f, -0.939199f},
    float3{0.000000f, 0.608439f, 0.793601f},
    float3{-0.333333f, -0.928397f, -0.164220f},
    float3{-0.666667f, 0.628898f, -0.400053f},
    float3{-1.000000f, -0.000000f, 0.000000f},
};

constexpr float3 kSpn1[8] = {
    float3{0.000000f, 1.000000f, 0.000000f}, float3{-0.516051f, 0.714286f, 0.472745f},
    float3{0.078990f, 0.428571f, -0.900048f}, float3{0.602198f, 0.142857f, 0.785461f},
    float3{-0.974614f, -0.142857f, -0.172395f}, float3{0.762340f, -0.428571f, -0.484938f},
    float3{-0.181685f, -0.714286f, 0.675860f}, float3{-0.000000f, -1.000000f, -0.000000f},
};

constexpr float3 kSpn2[16] = {
    float3{0.000000f, 0.000000f, 1.000000f}, float3{0.336994f, -0.367864f, 0.866667f},
    float3{-0.677266f, 0.059438f, 0.733333f}, float3{0.634881f, 0.486751f, 0.600000f},
    float3{-0.154052f, -0.870913f, 0.466667f}, float3{-0.506032f, 0.795500f, 0.333333f},
    float3{0.946204f, -0.254359f, 0.200000f}, float3{-0.885474f, -0.459882f, 0.066667f},
    float3{0.342275f, 0.937232f, -0.066667f}, float3{0.373847f, -0.905670f, -0.200000f},
    float3{-0.853934f, 0.399606f, -0.333333f}, float3{0.843895f, 0.264697f, -0.466667f},
    float3{-0.401126f, -0.692169f, -0.600000f}, float3{-0.145981f, 0.664012f, -0.733333f},
    float3{0.408121f, -0.286925f, -0.866667f}, float3{-0.000000f, -0.000000f, -1.000000f},
};

constexpr float3 kSpn3[16] = {
    float3{0.628945f, 0.674911f, -0.385907f}, float3{0.418351f, 0.439884f, -0.794660f},
    float3{0.926019f, 0.329212f, 0.184683f}, float3{-0.225970f, 0.925107f, -0.305147f},
    float3{0.672535f, -0.317517f, -0.668490f}, float3{0.324828f, 0.606254f, 0.725908f},
    float3{-0.470630f, 0.251300f, -0.845787f}, float3{0.815736f, -0.533206f, 0.224201f},
    float3{-0.575729f, 0.689246f, 0.439859f}, float3{-0.112624f, -0.630502f, -0.767974f},
    float3{0.276062f, -0.215587f, 0.936649f}, float3{-0.977336f, 0.119351f, -0.174841f},
    float3{0.122688f, -0.992256f, 0.019382f}, float3{-0.562275f, -0.092724f, 0.821736f},
    float3{-0.747574f, -0.653927f, -0.116242f}, float3{-0.628945f, -0.674911f, 0.385907f},
};

constexpr float3 kSpn4[16] = {
    float3{-0.513486f, -0.802627f, -0.303516f}, float3{-0.329417f, -0.930220f, 0.161785f},
    float3{-0.113287f, -0.521353f, -0.845788f}, float3{-0.950328f, -0.222046f, 0.218111f},
    float3{0.464981f, -0.885166f, 0.016571f}, float3{-0.521976f, 0.229310f, -0.821558f},
    float3{-0.345623f, -0.356336f, 0.868084f}, float3{0.701860f, -0.285260f, -0.652701f},
    float3{-0.803928f, 0.594684f, 0.007127f}, float3{0.593662f, -0.395610f, 0.700755f},
    float3{0.266437f, 0.545373f, -0.794719f}, float3{-0.335359f, 0.492208f, 0.803284f},
    float3{0.994251f, 0.088891f, 0.059699f}, float3{-0.041402f, 0.990302f, -0.132615f},
    float3{0.469425f, 0.505778f, 0.723761f}, float3{0.513486f, 0.802627f, 0.303516f},
};

constexpr float3 kSpVn5[32] = {
    float3{0.382081f, 0.918886f, -0.098303f}, float3{0.650970f, 0.595892f, 0.470267f},
    float3{0.550136f, -0.754869f, 0.357104f}, float3{-0.181620f, -0.226392f, 0.315110f},
    float3{0.896833f, 0.419476f, -0.138502f}, float3{-0.846077f, 0.475248f, 0.241414f},
    float3{-0.118985f, -0.729624f, -0.673418f}, float3{-0.202309f, 0.321186f, -0.001336f},
    float3{-0.387992f, 0.920914f, 0.037136f}, float3{-0.651299f, -0.241163f, -0.719479f},
    float3{-0.226884f, 0.258907f, -0.938876f}, float3{0.217251f, 0.314177f, 0.919230f},
    float3{0.823361f, -0.030952f, 0.566674f}, float3{-0.648360f, -0.662882f, 0.374455f},
    float3{0.292507f, 0.335440f, -0.491240f}, float3{-0.256196f, -0.179743f, 0.949766f},
    float3{0.353981f, -0.354754f, 0.865359f}, float3{0.543675f, -0.709539f, -0.448299f},
    float3{0.248602f, -0.236637f, -0.939250f}, float3{-0.072767f, -0.771426f, 0.632144f},
    float3{0.376908f, 0.055425f, 0.107197f}, float3{-0.152183f, 0.820610f, -0.550854f},
    float3{-0.762858f, 0.434625f, -0.478684f}, float3{-0.604668f, -0.753714f, -0.257473f},
    float3{0.800153f, -0.074823f, -0.595110f}, float3{0.047339f, 0.850153f, 0.524403f},
    float3{-0.789310f, -0.083667f, 0.608267f}, float3{-0.435870f, 0.459988f, 0.773575f},
    float3{0.940615f, -0.339424f, -0.005972f}, float3{-0.939158f, -0.149618f, -0.079873f},
    float3{0.017382f, -0.945689f, -0.034440f}, float3{-0.045413f, -0.211751f, -0.306299f},
};

constexpr float3 kSpVn6[32] = {
    float3{-0.727243f, -0.193675f, 0.567283f}, float3{-0.295083f, -0.828711f, 0.475567f},
    float3{-0.995703f, -0.076032f, -0.052861f}, float3{0.406178f, 0.627644f, 0.664141f},
    float3{0.453177f, -0.018639f, 0.891226f}, float3{0.034613f, 0.066587f, 0.389293f},
    float3{-0.211494f, 0.858032f, 0.468029f}, float3{-0.805961f, 0.499459f, 0.317753f},
    float3{-0.711991f, -0.228635f, -0.654137f}, float3{0.720579f, 0.535021f, -0.441042f},
    float3{0.373046f, -0.669333f, 0.642518f}, float3{0.683159f, -0.730082f, 0.016564f},
    float3{-0.386891f, 0.323358f, 0.863571f}, float3{0.948147f, -0.134710f, -0.287874f},
    float3{-0.129105f, 0.402723f, -0.050831f}, float3{0.437607f, -0.645013f, -0.626465f},
    float3{0.876379f, -0.234138f, 0.420880f}, float3{0.881941f, 0.419503f, 0.214937f},
    float3{-0.295534f, -0.763040f, -0.574830f}, float3{0.201295f, -0.357166f, 0.034116f},
    float3{0.090924f, -0.992554f, -0.081048f}, float3{-0.329153f, -0.143500f, -0.075175f},
    float3{0.110731f, 0.736193f, -0.667652f}, float3{0.527626f, 0.025130f, -0.849105f},
    float3{-0.721017f, -0.690451f, -0.058410f}, float3{-0.161251f, -0.320616f, 0.933383f},
    float3{-0.046937f, -0.277278f, -0.959643f}, float3{0.344410f, 0.938620f, 0.019314f},
    float3{0.127436f, 0.018955f, -0.397832f}, float3{-0.793211f, 0.464290f, -0.394019f},
    float3{-0.358985f, 0.309343f, -0.880589f}, float3{-0.337826f, 0.918535f, -0.205349f},
};

constexpr float3 kSpVn7[32] = {
    float3{-0.410460f, -0.591758f, -0.693790f}, float3{0.345327f, 0.801148f, 0.488785f},
    float3{-0.331667f, -0.940133f, -0.075232f}, float3{-0.862656f, -0.487171f, -0.126874f},
    float3{0.369731f, 0.254916f, 0.014690f}, float3{-0.744325f, 0.659225f, 0.106789f},
    float3{-0.053761f, -0.828511f, 0.557386f}, float3{-0.125478f, -0.240730f, 0.962237f},
    float3{0.062130f, -0.197417f, -0.351812f}, float3{-0.657577f, -0.492287f, 0.570303f},
    float3{0.353328f, 0.715205f, -0.602790f}, float3{-0.383008f, -0.047040f, 0.228408f},
    float3{0.816554f, -0.436954f, -0.377240f}, float3{0.557326f, -0.800918f, 0.218903f},
    float3{-0.818517f, 0.251103f, -0.505864f}, float3{0.829851f, 0.427577f, 0.358505f},
    float3{-0.194640f, 0.403027f, -0.118038f}, float3{0.177782f, -0.232706f, 0.295433f},
    float3{-0.253527f, 0.068370f, -0.947231f}, float3{0.017371f, 0.305274f, 0.491131f},
    float3{0.934119f, -0.225524f, 0.276239f}, float3{-0.260850f, 0.723730f, -0.638884f},
    float3{0.264240f, -0.861446f, -0.433692f}, float3{-0.555007f, 0.258905f, 0.790529f},
    float3{0.597648f, 0.800227f, -0.049532f}, float3{0.510981f, 0.224669f, 0.829338f},
    float3{-0.250464f, 0.888485f, 0.384352f}, float3{0.513797f, -0.376265f, 0.770998f},
    float3{0.539898f, 0.053152f, -0.840034f}, float3{0.947216f, 0.229984f, -0.223358f},
    float3{-0.966975f, 0.104162f, 0.232617f}, float3{0.021570f, 0.986682f, -0.159492f},
};

float3 hash31sphere() {
    const float3 rands = float3{hash1(), hash1(), hash1()};
    const float theta = 2.0f * 3.14159265358979f * rands.x;
    const float phi = acos(2.0f * rands.y - 1.0f);
    const float3 p = float3{cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)};
    return normalize(p);
}

float3 hash3box(float scale = 0.01f) {
    return hash31sphere() * cbrt(hash1()) * scale;
}

float3 GetFixedStencilPoint(int index) {
    if (index < 8) {
        return kSpn0[index];
    }
    if (index < 16) {
        return kSpn1[index - 8];
    }
    if (index < 32) {
        return kSpn2[index - 16];
    }
    if (index < 48) {
        return kSpn3[index - 32];
    }
    if (index < 64) {
        return kSpn4[index - 48];
    }
    if (index < 96) {
        return kSpVn5[index - 64];
    }
    if (index < 128) {
        return kSpVn6[index - 96];
    }
    return kSpVn7[index - 128];
}

float3 SphereRandom3Host(int index, float radius, float3 xMain, float3 yMain, float3 zMain) {
    const float3 local = GetFixedStencilPoint(index);
    return (xMain * local.x + yMain * local.y + zMain * local.z) * radius;
}

void CheckCudaOrDie(cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        std::cerr << context << ": " << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

float3x3 GetMatrixFromNormalHost(float3 v1) {
    v1 = normalize(v1);
    while (true) {
        const float3 r = UniformSampleSphere(float2{hash1(), hash1()});
        if (abs(dot(r, v1)) < 0.01f) {
            continue;
        }

        const float3 v2 = normalize(cross(v1, r));
        const float3 v3 = cross(v2, v1);
        return float3x3(v1, v2, v3);
    }
}

bool DetermineNextVertex(VolumeRender& volume, float epsilon, float g, float3 pos, float3 dir, float dis, float3* nextPos, float3* nextDir) {
    const float sMax = volume.max_density * epsilon;
    float t = 0.0f;
    int loopNum = 0;
    while (loopNum++ < 10000) {
        float rk = hash1();
        t -= log(1.0f - rk) / sMax;

        if (t > dis) {
            *nextPos = {0, 0, 0};
            *nextDir = {0, 0, 0};
            return false;
        }

        rk = hash1();
        const float density = volume.DensityAtPosition(0, pos + dir * t);
        const float s = density * epsilon;
        if (s / sMax > rk) {
            break;
        }
        if (density < 0.0f) {
            t -= density;
        }
    }

    *nextDir = SampleHenyeyGreenstein(hash1(), hash1(), dir, g);
    *nextPos = dir * t + pos;
    return true;
}

void MeanFreePathSample(VolumeRender& volume, std::vector<MRPNNSamplePoint>& samples, float3 ori, float3 dir, float3 lightDir, int maxCount, float epsilon, float g) {
    dir = normalize(dir);
    lightDir = normalize(lightDir);

    const float offset = RayBoxOffset(ori, dir);
    if (offset < 0.0f) {
        return;
    }

    float3 samplePosition = ori + dir * offset;
    float3 rayDirection = dir;
    for (int i = 0; i < 4; i++) {
        float3 nextPos;
        float3 nextDir;
        const float dis = RayBoxDistance(samplePosition, rayDirection);
        const bool inVolume = DetermineNextVertex(volume, epsilon, g, samplePosition, rayDirection, dis, &nextPos, &nextDir);

        if (!inVolume || static_cast<int>(samples.size()) >= maxCount) {
            return;
        }

        if (i == 0 || dot(samplePosition - nextPos, samplePosition - nextPos) > 1.0f / 64.0f || hash1() > 0.9f) {
            const float3 jitteredPos = hash1() > 0.5f ? nextPos + hash3box(1.0f / 128.0f) : nextPos;
            const float3 conditionedViewDir = hash1() > 0.25f ? dir : rayDirection;
            const float albedo = lerp(kMinZeta, kMaxZeta, hash1());
            samples.push_back(MRPNNSamplePoint{jitteredPos, conditionedViewDir, lightDir, epsilon, g, albedo});
        }

        samplePosition = nextPos;
        rayDirection = nextDir;
    }
}

void GetDesiredCountSample(
    VolumeRender& volume,
    std::vector<MRPNNSamplePoint>& samples,
    int count,
    float epsilon,
    float g,
    float3 lightDir) {
    samples.clear();
    int lastPrint = 0;
    const int printPer = std::max(1, count / 8);

    while (static_cast<int>(samples.size()) < count) {
        if (static_cast<int>(samples.size()) / printPer > lastPrint) {
            printf("Getting Samples: %.5f%%\n", static_cast<float>(samples.size()) / count * 100.0f);
            lastPrint = static_cast<int>(samples.size()) / printPer;
        }

        const float3 ori = hash31sphere() * 3.0f;
        const float3 dir = normalize(hash31sphere() + normalize(-ori));
        MeanFreePathSample(volume, samples, ori, dir, lightDir, count, epsilon, g);
    }

    if (static_cast<int>(samples.size()) > count) {
        samples.resize(count);
    }
}

MRPNNDescriptorRow BuildDescriptorRow(VolumeRender& volume, const MRPNNSamplePoint& sample, float predictRadiance, const BasisPair& basis) {
    MRPNNDescriptorRow row;
    row.g = sample.g;
    row.zetaPowAlpha = std::pow(sample.albedo, kAlbedoExponent);
    row.gamma = acos(dot(basis.main.x, basis.light.x));
    row.targetPredictRadiance = std::max(predictRadiance, 0.0f);

    int randIndex = 0;
    for (int i = 0; i < kStencilPointCount; i++) {
        Offset_Layer_ currentOffsetInfo = GetSamples23_(i);
        currentOffsetInfo.Layer += 0.1f;

        float3 currentOffsetPos;
        if (currentOffsetInfo.type >= 4) {
            currentOffsetPos = sample.position + SphereRandom3Host(
                randIndex,
                currentOffsetInfo.Offset,
                basis.main.x,
                basis.main.y,
                basis.main.z);
            randIndex++;
        } else {
            currentOffsetPos = sample.position + basis.light.x * currentOffsetInfo.Offset;
        }

        const int densityMip = static_cast<int>(currentOffsetInfo.Layer);
        row.density[i] = log(1.0f + sample.epsilon * volume.DensityAtPosition(densityMip, currentOffsetPos) / 64.0f);

        if (currentOffsetInfo.localindex == 0) {
            row.phase[i] = log(HenyeyGreenstein(dot(basis.main.x, basis.light.x), sample.g) + 1.0f);
        } else {
            const float3 msDir = normalize(currentOffsetPos - sample.position);
            const float radius = static_cast<float>(1 << densityMip) / 256.0f;
            const float angle = atan(0.5f * radius / currentOffsetInfo.Offset);
            const float hg0 = volume.GetHGLut(dot(basis.main.x, msDir), angle);
            const float hg1 = volume.GetHGLut(dot(basis.light.x, msDir), angle);
            row.phase[i] = log(hg0 * hg1 + 1.0f);
        }

        const int trMip = std::max(densityMip - 1, 0);
        row.transmittance[i] = volume.TrAtPosition(trMip, currentOffsetPos, basis.light.x);
    }

    return row;
}

MRPNNDescriptorRow BuildDescriptorRow(VolumeRender& volume, const MRPNNSamplePoint& sample, float predictRadiance) {
    BasisPair basis = {GetMatrixFromNormalHost(sample.viewDir), GetMatrixFromNormalHost(sample.lightDir)};
    return BuildDescriptorRow(volume, sample, predictRadiance, basis);
}

__global__ void ExtractMRPNNDescriptorKernel(
    const MRPNNSamplePoint* samples,
    const BasisPair* bases,
    DeviceDescriptorRow* rows,
    int sampleCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sampleCount) {
        return;
    }

    const MRPNNSamplePoint sample = samples[idx];
    const BasisPair basis = bases[idx];

    DeviceDescriptorRow row = {};
    row.gamma = acos(dot(basis.main.x, basis.light.x));

    int randIndex = 0;
    for (int i = 0; i < kStencilPointCount; i++) {
        Offset_Layer_ currentOffsetInfo = GetSamples23_(i);
        currentOffsetInfo.Layer += 0.1f;

        float3 currentOffsetPos;
        if (currentOffsetInfo.type >= 4) {
            currentOffsetPos = sample.position + SphereRandom3(
                randIndex,
                currentOffsetInfo.Offset,
                basis.main.x,
                basis.main.y,
                basis.main.z,
                sample.g);
            randIndex++;
        } else {
            currentOffsetPos = sample.position + basis.light.x * currentOffsetInfo.Offset;
        }

        const int densityMip = static_cast<int>(currentOffsetInfo.Layer);
        row.density[i] = log(1.0f + sample.epsilon * MipDensityDynamic(densityMip, currentOffsetPos) / 64.0f);

        if (currentOffsetInfo.localindex == 0) {
            row.phase[i] = log(HenyeyGreenstein(dot(basis.main.x, basis.light.x), sample.g) + 1.0f);
        } else {
            const float3 msDir = normalize(currentOffsetPos - sample.position);
            const float radius = static_cast<float>(1 << densityMip) / 256.0f;
            const float angle = atan(0.5f * radius / currentOffsetInfo.Offset);
            const float u0 = dot(basis.main.x, msDir) * 0.5f + 0.5f;
            const float u1 = dot(basis.light.x, msDir) * 0.5f + 0.5f;
            const float v = angle / (3.1415926535f * 60.0f / 180.0f);
            const float hg0 = tex2D<float>(_HGLut, u0, v);
            const float hg1 = tex2D<float>(_HGLut, u1, v);
            row.phase[i] = log(hg0 * hg1 + 1.0f);
        }

        row.transmittance[i] = ShadowTerm_TRTex(
            currentOffsetPos,
            basis.light.x,
            basis.main.x,
            float3{1.0f, 1.0f, 1.0f},
            sample.g,
            max(densityMip - 1, 0)).x;
    }

    rows[idx] = row;
}

void AppendGpuDescriptorRows(
    VolumeRender& volume,
    const std::vector<MRPNNSamplePoint>& samples,
    const std::vector<float3>& predictRadiances,
    std::vector<MRPNNDescriptorRow>& rows) {
    if (samples.empty()) {
        return;
    }
    if (samples.size() != predictRadiances.size()) {
        std::cerr << "Descriptor extraction failed: sample/target count mismatch.\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<BasisPair> bases(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        bases[i] = BasisPair{GetMatrixFromNormalHost(samples[i].viewDir), GetMatrixFromNormalHost(samples[i].lightDir)};
    }

    MRPNNSamplePoint* deviceSamples = nullptr;
    BasisPair* deviceBases = nullptr;
    DeviceDescriptorRow* deviceRows = nullptr;
    CheckCudaOrDie(cudaMalloc(&deviceSamples, sizeof(MRPNNSamplePoint) * samples.size()), "cudaMalloc(deviceSamples)");
    CheckCudaOrDie(cudaMalloc(&deviceBases, sizeof(BasisPair) * bases.size()), "cudaMalloc(deviceBases)");
    CheckCudaOrDie(cudaMalloc(&deviceRows, sizeof(DeviceDescriptorRow) * samples.size()), "cudaMalloc(deviceRows)");

    CheckCudaOrDie(cudaMemcpy(deviceSamples, samples.data(), sizeof(MRPNNSamplePoint) * samples.size(), cudaMemcpyHostToDevice), "cudaMemcpy(samples)");
    CheckCudaOrDie(cudaMemcpy(deviceBases, bases.data(), sizeof(BasisPair) * bases.size(), cudaMemcpyHostToDevice), "cudaMemcpy(bases)");

    for (size_t i = 0; i < samples.size(); i++) {
        volume.UpdateHGLut(samples[i].g);
        ExtractMRPNNDescriptorKernel<<<1, 1>>>(
            deviceSamples + i,
            deviceBases + i,
            deviceRows + i,
            1);
        CheckCudaOrDie(cudaGetLastError(), "ExtractMRPNNDescriptorKernel launch");
        CheckCudaOrDie(cudaDeviceSynchronize(), "ExtractMRPNNDescriptorKernel sync");
    }

    std::vector<DeviceDescriptorRow> gpuRows(samples.size());
    CheckCudaOrDie(cudaMemcpy(gpuRows.data(), deviceRows, sizeof(DeviceDescriptorRow) * samples.size(), cudaMemcpyDeviceToHost), "cudaMemcpy(deviceRows)");

    cudaFree(deviceSamples);
    cudaFree(deviceBases);
    cudaFree(deviceRows);

    rows.reserve(rows.size() + samples.size());
    for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
        MRPNNDescriptorRow row;
        const DeviceDescriptorRow& gpu = gpuRows[sampleIndex];
        for (int featureIndex = 0; featureIndex < kStencilPointCount; featureIndex++) {
            row.density[featureIndex] = gpu.density[featureIndex];
            row.transmittance[featureIndex] = gpu.transmittance[featureIndex];
            row.phase[featureIndex] = gpu.phase[featureIndex];
        }
        row.g = samples[sampleIndex].g;
        row.zetaPowAlpha = std::pow(samples[sampleIndex].albedo, kAlbedoExponent);
        row.gamma = gpu.gamma;
        row.targetPredictRadiance = std::max(predictRadiances[sampleIndex].x, 0.0f);
        rows.push_back(row);
    }
}

bool RunDescriptorValidation(const std::string& volumePath, const std::string& volumeName, int sampleCount) {
    std::cout << "Validating descriptor parity for " << volumeName << "\n";

    VolumeRender volume(volumePath);
    const float3 lightDir = hash31sphere();
    const float epsilon = lerp(kEpsilonMin, kEpsilonMax, hash1());
    const float g = lerp(0.0f, kMaxG, hash1());

    volume.UpdateHGLut(g);
    volume.Update_TR(lightDir, epsilon, true);

    std::vector<MRPNNSamplePoint> samples;
    GetDesiredCountSample(volume, samples, sampleCount, epsilon, g, lightDir);
    if (samples.empty()) {
        std::cerr << "Validation failed: no samples generated.\n";
        return false;
    }

    std::vector<BasisPair> bases(samples.size());
    std::vector<MRPNNDescriptorRow> hostRows;
    hostRows.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        bases[i] = BasisPair{GetMatrixFromNormalHost(samples[i].viewDir), GetMatrixFromNormalHost(samples[i].lightDir)};
        hostRows.push_back(BuildDescriptorRow(volume, samples[i], 0.0f, bases[i]));
    }

    MRPNNSamplePoint* deviceSamples = nullptr;
    BasisPair* deviceBases = nullptr;
    DeviceDescriptorRow* deviceRows = nullptr;
    CheckCudaOrDie(cudaMalloc(&deviceSamples, sizeof(MRPNNSamplePoint) * samples.size()), "cudaMalloc(deviceSamples)");
    CheckCudaOrDie(cudaMalloc(&deviceBases, sizeof(BasisPair) * bases.size()), "cudaMalloc(deviceBases)");
    CheckCudaOrDie(cudaMalloc(&deviceRows, sizeof(DeviceDescriptorRow) * samples.size()), "cudaMalloc(deviceRows)");

    CheckCudaOrDie(cudaMemcpy(deviceSamples, samples.data(), sizeof(MRPNNSamplePoint) * samples.size(), cudaMemcpyHostToDevice), "cudaMemcpy(samples)");
    CheckCudaOrDie(cudaMemcpy(deviceBases, bases.data(), sizeof(BasisPair) * bases.size(), cudaMemcpyHostToDevice), "cudaMemcpy(bases)");

    const int threads = 64;
    const int blocks = static_cast<int>((samples.size() + threads - 1) / threads);
    ExtractMRPNNDescriptorKernel<<<blocks, threads>>>(deviceSamples, deviceBases, deviceRows, static_cast<int>(samples.size()));
    CheckCudaOrDie(cudaGetLastError(), "ExtractMRPNNDescriptorKernel launch");
    CheckCudaOrDie(cudaDeviceSynchronize(), "ExtractMRPNNDescriptorKernel sync");

    std::vector<DeviceDescriptorRow> gpuRows(samples.size());
    CheckCudaOrDie(cudaMemcpy(gpuRows.data(), deviceRows, sizeof(DeviceDescriptorRow) * samples.size(), cudaMemcpyDeviceToHost), "cudaMemcpy(deviceRows)");

    cudaFree(deviceSamples);
    cudaFree(deviceBases);
    cudaFree(deviceRows);

    float maxDensityDiff = 0.0f;
    float maxTransmittanceDiff = 0.0f;
    float maxPhaseDiff = 0.0f;
    float maxGammaDiff = 0.0f;
    float phaseDiffSum = 0.0f;
    int densitySampleIndex = -1;
    int transmittanceSampleIndex = -1;
    int phaseSampleIndex = -1;
    int gammaSampleIndex = -1;
    int densityFeatureIndex = -1;
    int transmittanceFeatureIndex = -1;
    int phaseFeatureIndex = -1;
    std::vector<float> phaseDiffs;
    phaseDiffs.reserve(samples.size() * kStencilPointCount);

    for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
        const MRPNNDescriptorRow& host = hostRows[sampleIndex];
        const DeviceDescriptorRow& gpu = gpuRows[sampleIndex];

        const float gammaDiff = std::abs(host.gamma - gpu.gamma);
        if (gammaDiff > maxGammaDiff) {
            maxGammaDiff = gammaDiff;
            gammaSampleIndex = static_cast<int>(sampleIndex);
        }

        for (int featureIndex = 0; featureIndex < kStencilPointCount; featureIndex++) {
            const float densityDiff = std::abs(host.density[featureIndex] - gpu.density[featureIndex]);
            if (densityDiff > maxDensityDiff) {
                maxDensityDiff = densityDiff;
                densitySampleIndex = static_cast<int>(sampleIndex);
                densityFeatureIndex = featureIndex;
            }

            const float transmittanceDiff = std::abs(host.transmittance[featureIndex] - gpu.transmittance[featureIndex]);
            if (transmittanceDiff > maxTransmittanceDiff) {
                maxTransmittanceDiff = transmittanceDiff;
                transmittanceSampleIndex = static_cast<int>(sampleIndex);
                transmittanceFeatureIndex = featureIndex;
            }

            const float phaseDiff = std::abs(host.phase[featureIndex] - gpu.phase[featureIndex]);
            phaseDiffSum += phaseDiff;
            phaseDiffs.push_back(phaseDiff);
            if (phaseDiff > maxPhaseDiff) {
                maxPhaseDiff = phaseDiff;
                phaseSampleIndex = static_cast<int>(sampleIndex);
                phaseFeatureIndex = featureIndex;
            }
        }
    }

    const float meanPhaseDiff = phaseDiffs.empty() ? 0.0f : phaseDiffSum / static_cast<float>(phaseDiffs.size());
    float p95PhaseDiff = 0.0f;
    if (!phaseDiffs.empty()) {
        const size_t p95Index = static_cast<size_t>(0.95f * static_cast<float>(phaseDiffs.size() - 1));
        std::nth_element(phaseDiffs.begin(), phaseDiffs.begin() + p95Index, phaseDiffs.end());
        p95PhaseDiff = phaseDiffs[p95Index];
    }

    std::cout << "  condition: epsilon=" << epsilon
              << " g=" << g
              << " lightDir=(" << lightDir.x << ", " << lightDir.y << ", " << lightDir.z << ")\n";
    std::cout << "  density max abs diff: " << maxDensityDiff
              << " at sample " << densitySampleIndex
              << ", feature " << densityFeatureIndex << "\n";
    std::cout << "  transmittance max abs diff: " << maxTransmittanceDiff
              << " at sample " << transmittanceSampleIndex
              << ", feature " << transmittanceFeatureIndex << "\n";
    std::cout << "  phase max abs diff: " << maxPhaseDiff
              << " at sample " << phaseSampleIndex
              << ", feature " << phaseFeatureIndex << "\n";
    std::cout << "  phase mean abs diff: " << meanPhaseDiff << "\n";
    std::cout << "  phase p95 abs diff: " << p95PhaseDiff << "\n";
    std::cout << "  gamma max abs diff: " << maxGammaDiff
              << " at sample " << gammaSampleIndex << "\n";

    constexpr float kDensityTolerance = 2e-2f;
    constexpr float kTransmittanceTolerance = 5e-3f;
    constexpr float kPhaseTolerance = 5e-3f;
    constexpr float kGammaTolerance = 1e-5f;
    const bool passed =
        maxDensityDiff <= kDensityTolerance &&
        maxTransmittanceDiff <= kTransmittanceTolerance &&
        maxPhaseDiff <= kPhaseTolerance &&
        maxGammaDiff <= kGammaTolerance;

    std::cout << "  result: " << (passed ? "PASS" : "FAIL") << "\n";
    if (!passed && maxPhaseDiff > kPhaseTolerance && meanPhaseDiff < 5e-3f && p95PhaseDiff < 1e-2f) {
        std::cout << "  note: phase mismatch looks localized; mean/p95 stay small while max spikes.\n";
    }
    return passed;
}

struct DiffStats {
    std::vector<float> values;
    double sum = 0.0;
    float maxValue = 0.0f;
    int maxSampleIndex = -1;
    int maxFeatureIndex = -1;

    void add(float value, int sampleIndex, int featureIndex) {
        values.push_back(value);
        sum += value;
        if (value > maxValue) {
            maxValue = value;
            maxSampleIndex = sampleIndex;
            maxFeatureIndex = featureIndex;
        }
    }

    float mean() const {
        return values.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(values.size()));
    }

    float percentile(float p) {
        if (values.empty()) {
            return 0.0f;
        }
        const size_t index = static_cast<size_t>(std::clamp(p, 0.0f, 1.0f) * static_cast<float>(values.size() - 1));
        std::nth_element(values.begin(), values.begin() + index, values.end());
        return values[index];
    }

    void print(const char* name) {
        const float p95 = percentile(0.95f);
        const float p99 = percentile(0.99f);
        std::cout << "  " << name
                  << " mean=" << mean()
                  << " p95=" << p95
                  << " p99=" << p99
                  << " max=" << maxValue
                  << " at sample " << maxSampleIndex;
        if (maxFeatureIndex >= 0) {
            std::cout << ", feature " << maxFeatureIndex;
        }
        std::cout << "\n";
    }
};

bool RunSkewValidation(const std::string& volumePath, const std::string& volumeName, int sampleCount) {
    std::cout << "Validating host/GPU descriptor skew for " << volumeName << "\n";

    VolumeRender volume(volumePath);
    const float3 lightDir = hash31sphere();
    const float epsilon = lerp(kEpsilonMin, kEpsilonMax, hash1());
    const float g = lerp(0.0f, kMaxG, hash1());

    volume.UpdateHGLut(g);
    volume.Update_TR(lightDir, epsilon, true);

    std::vector<MRPNNSamplePoint> samples;
    GetDesiredCountSample(volume, samples, sampleCount, epsilon, g, lightDir);
    if (samples.empty()) {
        std::cerr << "Skew validation failed: no samples generated.\n";
        return false;
    }

    std::vector<BasisPair> bases(samples.size());
    std::vector<MRPNNDescriptorRow> hostRows;
    hostRows.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        bases[i] = BasisPair{GetMatrixFromNormalHost(samples[i].viewDir), GetMatrixFromNormalHost(samples[i].lightDir)};
        hostRows.push_back(BuildDescriptorRow(volume, samples[i], 0.0f, bases[i]));
    }

    MRPNNSamplePoint* deviceSamples = nullptr;
    BasisPair* deviceBases = nullptr;
    DeviceDescriptorRow* deviceRows = nullptr;
    CheckCudaOrDie(cudaMalloc(&deviceSamples, sizeof(MRPNNSamplePoint) * samples.size()), "cudaMalloc(deviceSamples)");
    CheckCudaOrDie(cudaMalloc(&deviceBases, sizeof(BasisPair) * bases.size()), "cudaMalloc(deviceBases)");
    CheckCudaOrDie(cudaMalloc(&deviceRows, sizeof(DeviceDescriptorRow) * samples.size()), "cudaMalloc(deviceRows)");

    CheckCudaOrDie(cudaMemcpy(deviceSamples, samples.data(), sizeof(MRPNNSamplePoint) * samples.size(), cudaMemcpyHostToDevice), "cudaMemcpy(samples)");
    CheckCudaOrDie(cudaMemcpy(deviceBases, bases.data(), sizeof(BasisPair) * bases.size(), cudaMemcpyHostToDevice), "cudaMemcpy(bases)");

    const int threads = 64;
    const int blocks = static_cast<int>((samples.size() + threads - 1) / threads);
    ExtractMRPNNDescriptorKernel<<<blocks, threads>>>(deviceSamples, deviceBases, deviceRows, static_cast<int>(samples.size()));
    CheckCudaOrDie(cudaGetLastError(), "ExtractMRPNNDescriptorKernel launch");
    CheckCudaOrDie(cudaDeviceSynchronize(), "ExtractMRPNNDescriptorKernel sync");

    std::vector<DeviceDescriptorRow> gpuRows(samples.size());
    CheckCudaOrDie(cudaMemcpy(gpuRows.data(), deviceRows, sizeof(DeviceDescriptorRow) * samples.size(), cudaMemcpyDeviceToHost), "cudaMemcpy(deviceRows)");

    cudaFree(deviceSamples);
    cudaFree(deviceBases);
    cudaFree(deviceRows);

    DiffStats densityStats;
    DiffStats transmittanceStats;
    DiffStats phaseStats;
    DiffStats gammaStats;
    densityStats.values.reserve(samples.size() * kStencilPointCount);
    transmittanceStats.values.reserve(samples.size() * kStencilPointCount);
    phaseStats.values.reserve(samples.size() * kStencilPointCount);
    gammaStats.values.reserve(samples.size());

    for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
        const MRPNNDescriptorRow& host = hostRows[sampleIndex];
        const DeviceDescriptorRow& gpu = gpuRows[sampleIndex];
        gammaStats.add(std::abs(host.gamma - gpu.gamma), static_cast<int>(sampleIndex), -1);
        for (int featureIndex = 0; featureIndex < kStencilPointCount; featureIndex++) {
            densityStats.add(std::abs(host.density[featureIndex] - gpu.density[featureIndex]), static_cast<int>(sampleIndex), featureIndex);
            transmittanceStats.add(std::abs(host.transmittance[featureIndex] - gpu.transmittance[featureIndex]), static_cast<int>(sampleIndex), featureIndex);
            phaseStats.add(std::abs(host.phase[featureIndex] - gpu.phase[featureIndex]), static_cast<int>(sampleIndex), featureIndex);
        }
    }

    std::cout << "  condition: epsilon=" << epsilon
              << " g=" << g
              << " lightDir=(" << lightDir.x << ", " << lightDir.y << ", " << lightDir.z << ")"
              << " samples=" << samples.size() << "\n";
    densityStats.print("density abs diff");
    transmittanceStats.print("transmittance abs diff");
    phaseStats.print("phase abs diff");
    gammaStats.print("gamma abs diff");

    const bool lowSkew =
        densityStats.percentile(0.95f) <= 2e-2f &&
        transmittanceStats.percentile(0.95f) <= 5e-3f &&
        phaseStats.percentile(0.95f) <= 1e-2f &&
        gammaStats.maxValue <= 1e-5f;
    std::cout << "  feature-skew risk: " << (lowSkew ? "LOW" : "NEEDS MODEL-SENSITIVITY CHECK") << "\n";
    return lowSkew;
}

bool RunTargetValidation(const std::string& volumePath, const std::string& volumeName, int sampleCount) {
    std::cout << "Validating target definition for " << volumeName << "\n";

    VolumeRender volume(volumePath);
    const float3 lightDir = hash31sphere();
    const float epsilon = lerp(kEpsilonMin, kEpsilonMax, hash1());
    const float g = lerp(0.0f, kMaxG, hash1());

    std::vector<MRPNNSamplePoint> samples;
    GetDesiredCountSample(volume, samples, sampleCount, epsilon, g, lightDir);
    if (samples.empty()) {
        std::cerr << "Target validation failed: no samples generated.\n";
        return false;
    }

    std::vector<float3> samplePositions;
    std::vector<float3> sampleDirs;
    std::vector<float3> sampleLightDirs;
    std::vector<float> sampleEpsilons;
    std::vector<float> sampleGs;
    std::vector<float> sampleScatters;
    samplePositions.reserve(samples.size());
    sampleDirs.reserve(samples.size());
    sampleLightDirs.reserve(samples.size());
    sampleEpsilons.reserve(samples.size());
    sampleGs.reserve(samples.size());
    sampleScatters.reserve(samples.size());

    for (const MRPNNSamplePoint& sample : samples) {
        samplePositions.push_back(sample.position);
        sampleDirs.push_back(sample.viewDir);
        sampleLightDirs.push_back(sample.lightDir);
        sampleEpsilons.push_back(sample.epsilon);
        sampleGs.push_back(sample.g);
        sampleScatters.push_back(sample.albedo);
    }

    const float3 lightColor = {1.0f, 1.0f, 1.0f};
    const std::vector<float3> predictRadiances = volume.GetSamples(
        sampleEpsilons,
        samplePositions,
        sampleDirs,
        sampleLightDirs,
        sampleGs,
        sampleScatters,
        lightColor,
        kPathTraceScatterCount,
        kPathTraceSampleCount);

    float maxPredictMismatch = 0.0f;
    float maxDirectContribution = 0.0f;
    float maxTotalGap = 0.0f;
    float avgPredict = 0.0f;
    float avgDirect = 0.0f;

    for (size_t i = 0; i < samples.size(); i++) {
        const float predict = predictRadiances[i].x;
        const float direct = volume.GetTr(
            samples[i].position,
            samples[i].viewDir,
            samples[i].lightDir,
            samples[i].epsilon,
            samples[i].g,
            512).x * samples[i].albedo;
        const MRPNNDescriptorRow row = BuildDescriptorRow(volume, samples[i], predict);

        maxPredictMismatch = std::max(maxPredictMismatch, std::abs(row.targetPredictRadiance - predict));
        maxDirectContribution = std::max(maxDirectContribution, direct);
        maxTotalGap = std::max(maxTotalGap, std::abs((predict + direct) - row.targetPredictRadiance));
        avgPredict += predict;
        avgDirect += direct;
    }

    avgPredict /= static_cast<float>(samples.size());
    avgDirect /= static_cast<float>(samples.size());

    std::cout << "  condition: epsilon=" << epsilon
              << " g=" << g
              << " lightDir=(" << lightDir.x << ", " << lightDir.y << ", " << lightDir.z << ")\n";
    std::cout << "  avg predict target: " << avgPredict << "\n";
    std::cout << "  avg direct term: " << avgDirect << "\n";
    std::cout << "  max(target - predict): " << maxPredictMismatch << "\n";
    std::cout << "  max((predict + direct) - target): " << maxTotalGap << "\n";

    const bool informativeDirect = maxDirectContribution > 1e-4f;
    const bool matchesPredict = maxPredictMismatch <= 1e-6f;
    const bool excludesDirect = maxTotalGap > 1e-4f;
    const bool passed = informativeDirect && matchesPredict && excludesDirect;

    std::cout << "  result: " << (passed ? "PASS" : "FAIL") << "\n";
    return passed;
}

void WriteRow(std::ofstream& outfile, const MRPNNDescriptorRow& row) {
    for (float value : row.density) {
        outfile << std::setiosflags(std::ios::fixed) << value << ",";
    }
    for (float value : row.transmittance) {
        outfile << std::setiosflags(std::ios::fixed) << value << ",";
    }
    for (float value : row.phase) {
        outfile << std::setiosflags(std::ios::fixed) << value << ",";
    }
    outfile << std::setiosflags(std::ios::fixed) << row.g << ",";
    outfile << std::setiosflags(std::ios::fixed) << row.zetaPowAlpha << ",";
    outfile << std::setiosflags(std::ios::fixed) << row.gamma << ",";
    outfile << std::setiosflags(std::ios::fixed) << row.targetPredictRadiance << "\n";
}

void DumpPairedDescriptorsForVolume(
    const std::string& volumePath,
    const std::string& volumeName,
    int sampleCount,
    std::ofstream& hostOutfile,
    std::ofstream& gpuOutfile) {
    std::cout << "Dumping paired descriptors for " << volumeName << "\n";

    VolumeRender volume(volumePath);
    const float3 lightDir = hash31sphere();
    const float epsilon = lerp(kEpsilonMin, kEpsilonMax, hash1());

    volume.Update_TR(lightDir, epsilon, true);

    std::vector<MRPNNSamplePoint> samples;
    samples.reserve(sampleCount);
    while (static_cast<int>(samples.size()) < sampleCount) {
        const float g = lerp(0.0f, kMaxG, hash1());
        std::vector<MRPNNSamplePoint> oneSample;
        GetDesiredCountSample(volume, oneSample, 1, epsilon, g, lightDir);
        if (!oneSample.empty()) {
            samples.push_back(oneSample.front());
        }
    }

    std::vector<float3> samplePositions;
    std::vector<float3> sampleDirs;
    std::vector<float3> sampleLightDirs;
    std::vector<float> sampleEpsilons;
    std::vector<float> sampleGs;
    std::vector<float> sampleScatters;
    samplePositions.reserve(samples.size());
    sampleDirs.reserve(samples.size());
    sampleLightDirs.reserve(samples.size());
    sampleEpsilons.reserve(samples.size());
    sampleGs.reserve(samples.size());
    sampleScatters.reserve(samples.size());
    for (const MRPNNSamplePoint& sample : samples) {
        samplePositions.push_back(sample.position);
        sampleDirs.push_back(sample.viewDir);
        sampleLightDirs.push_back(sample.lightDir);
        sampleEpsilons.push_back(sample.epsilon);
        sampleGs.push_back(sample.g);
        sampleScatters.push_back(sample.albedo);
    }

    const float3 lightColor = {1.0f, 1.0f, 1.0f};
    const std::vector<float3> predictRadiances = volume.GetSamples(
        sampleEpsilons,
        samplePositions,
        sampleDirs,
        sampleLightDirs,
        sampleGs,
        sampleScatters,
        lightColor,
        kPathTraceScatterCount,
        kPathTraceSampleCount);

    std::vector<BasisPair> bases(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        bases[i] = BasisPair{GetMatrixFromNormalHost(samples[i].viewDir), GetMatrixFromNormalHost(samples[i].lightDir)};
    }

    MRPNNSamplePoint* deviceSamples = nullptr;
    BasisPair* deviceBases = nullptr;
    DeviceDescriptorRow* deviceRows = nullptr;
    CheckCudaOrDie(cudaMalloc(&deviceSamples, sizeof(MRPNNSamplePoint) * samples.size()), "cudaMalloc(deviceSamples)");
    CheckCudaOrDie(cudaMalloc(&deviceBases, sizeof(BasisPair) * bases.size()), "cudaMalloc(deviceBases)");
    CheckCudaOrDie(cudaMalloc(&deviceRows, sizeof(DeviceDescriptorRow) * samples.size()), "cudaMalloc(deviceRows)");

    CheckCudaOrDie(cudaMemcpy(deviceSamples, samples.data(), sizeof(MRPNNSamplePoint) * samples.size(), cudaMemcpyHostToDevice), "cudaMemcpy(samples)");
    CheckCudaOrDie(cudaMemcpy(deviceBases, bases.data(), sizeof(BasisPair) * bases.size(), cudaMemcpyHostToDevice), "cudaMemcpy(bases)");

    for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
        volume.UpdateHGLut(samples[sampleIndex].g);

        const MRPNNDescriptorRow hostRow = BuildDescriptorRow(
            volume,
            samples[sampleIndex],
            predictRadiances[sampleIndex].x,
            bases[sampleIndex]);
        WriteRow(hostOutfile, hostRow);

        ExtractMRPNNDescriptorKernel<<<1, 1>>>(
            deviceSamples + sampleIndex,
            deviceBases + sampleIndex,
            deviceRows + sampleIndex,
            1);
        CheckCudaOrDie(cudaGetLastError(), "ExtractMRPNNDescriptorKernel launch");
        CheckCudaOrDie(cudaDeviceSynchronize(), "ExtractMRPNNDescriptorKernel sync");

        DeviceDescriptorRow gpu = {};
        CheckCudaOrDie(cudaMemcpy(&gpu, deviceRows + sampleIndex, sizeof(DeviceDescriptorRow), cudaMemcpyDeviceToHost), "cudaMemcpy(one device row)");

        MRPNNDescriptorRow gpuRow;
        for (int featureIndex = 0; featureIndex < kStencilPointCount; featureIndex++) {
            gpuRow.density[featureIndex] = gpu.density[featureIndex];
            gpuRow.transmittance[featureIndex] = gpu.transmittance[featureIndex];
            gpuRow.phase[featureIndex] = gpu.phase[featureIndex];
        }
        gpuRow.g = samples[sampleIndex].g;
        gpuRow.zetaPowAlpha = std::pow(samples[sampleIndex].albedo, kAlbedoExponent);
        gpuRow.gamma = gpu.gamma;
        gpuRow.targetPredictRadiance = std::max(predictRadiances[sampleIndex].x, 0.0f);
        WriteRow(gpuOutfile, gpuRow);
    }

    cudaFree(deviceSamples);
    cudaFree(deviceBases);
    cudaFree(deviceRows);

    std::cout << "  dumped rows: " << samples.size()
              << " epsilon=" << epsilon
              << " lightDir=(" << lightDir.x << ", " << lightDir.y << ", " << lightDir.z << ")\n";
}

namespace fs = std::filesystem;

static void CollectVolFiles(const fs::path& dir, std::vector<std::string>& list)
{
    list.clear();

    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;

        const auto& p = e.path();
        if (p.extension() == ".vol") {
            // 如果你后续代码期望相对文件名：用 filename()
            list.push_back(p.filename().string());

            // 如果你后续加载需要完整路径：改成这一行
            // list.push_back(p.string());
        }
    }

    std::sort(list.begin(), list.end());
}

} // namespace

int main(int argc, char** argv) {
    unsigned randomSeed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    bool useFixedSeed = false;
    bool validateOnly = false;
    bool validateAll = false;
    bool validateTarget = false;
    bool validateSkew = false;
    bool validateSkewAll = false;
    bool dumpPairedDescriptors = false;
    bool dumpPairedDescriptorsAll = false;
    int skewSampleCount = 128;
    for (int argIndex = 1; argIndex < argc; argIndex++) {
        const std::string arg = argv[argIndex];
        if (arg == "--seed" && argIndex + 1 < argc) {
            randomSeed = static_cast<unsigned>(std::strtoul(argv[argIndex + 1], nullptr, 10));
            useFixedSeed = true;
            argIndex++;
        } else if (arg == "--validate") {
            validateOnly = true;
        } else if (arg == "--validate-all") {
            validateAll = true;
        } else if (arg == "--validate-target") {
            validateTarget = true;
        } else if (arg == "--validate-skew") {
            validateSkew = true;
        } else if (arg == "--validate-skew-all") {
            validateSkewAll = true;
        } else if (arg == "--dump-paired-descriptors") {
            dumpPairedDescriptors = true;
        } else if (arg == "--dump-paired-descriptors-all") {
            dumpPairedDescriptorsAll = true;
        } else if (arg == "--skew-samples" && argIndex + 1 < argc) {
            skewSampleCount = std::max(1, std::atoi(argv[argIndex + 1]));
            argIndex++;
        }
    }
    srand(randomSeed);

    const std::string dataPath = "D:/Course/Projects/HairRender/MRPNN/Data/";
    const std::string dataName = "DS_MRPNN_paper_" + std::to_string(kSamplesPerModel) + "_per_model.csv";
    const std::string relativePath = "D:/Course/Projects/HairRender/MRPNN/MyData/mrpnn/";

    std::vector<std::string> dataList;
    //= {
    //    "dense.512.txt",
    //    "mediocris_high.512.txt",
    //    "cumulus_humilis.512.txt",
    //    "cumulus_congestus1.512.txt",
    //};

    CollectVolFiles(relativePath, dataList);
    for (auto& s : dataList) {
        std::cout << s << std::endl;
    }
    if (dataList.empty()) {
        std::cerr << "No .vol files found in " << relativePath << "\n";
        return 1;
    }

    if (validateOnly || validateAll) {
        if (useFixedSeed) {
            std::cout << "Using fixed seed: " << randomSeed << "\n";
        }
        bool allPassed = true;
        const int volumeCount = validateAll ? static_cast<int>(dataList.size()) : std::min<int>(1, dataList.size());
        for (int volumeIndex = 0; volumeIndex < volumeCount; volumeIndex++) {
            const std::string& volumeName = dataList[volumeIndex];
            allPassed = RunDescriptorValidation(relativePath + volumeName, volumeName, 8) && allPassed;
        }
        return allPassed ? 0 : 1;
    }
    if (validateTarget) {
        if (useFixedSeed) {
            std::cout << "Using fixed seed: " << randomSeed << "\n";
        }
        if (dataList.empty()) {
            return 1;
        }
        return RunTargetValidation(relativePath + dataList[0], dataList[0], 8) ? 0 : 1;
    }
    if (validateSkew || validateSkewAll) {
        if (useFixedSeed) {
            std::cout << "Using fixed seed: " << randomSeed << "\n";
        }
        bool allPassed = true;
        const int volumeCount = validateSkewAll ? static_cast<int>(dataList.size()) : std::min<int>(1, dataList.size());
        for (int volumeIndex = 0; volumeIndex < volumeCount; volumeIndex++) {
            const std::string& volumeName = dataList[volumeIndex];
            allPassed = RunSkewValidation(relativePath + volumeName, volumeName, skewSampleCount) && allPassed;
        }
        return allPassed ? 0 : 1;
    }
    if (dumpPairedDescriptors || dumpPairedDescriptorsAll) {
        if (useFixedSeed) {
            std::cout << "Using fixed seed: " << randomSeed << "\n";
        }
        const std::string hostPath = dataPath + "skew_host.csv";
        const std::string gpuPath = dataPath + "skew_gpu.csv";
        std::ofstream hostOutfile(hostPath);
        std::ofstream gpuOutfile(gpuPath);
        hostOutfile << "# columns=" << kOutputColumns
                    << ", layout=F_mu[192],F_S[192],F_P[192],g,zeta_pow_alpha,gamma,target_predict_radiance"
                    << ", source=host"
                    << ", paired_descriptor_test=true"
                    << ", samples_per_volume=" << skewSampleCount
                    << "\n";
        gpuOutfile << "# columns=" << kOutputColumns
                   << ", layout=F_mu[192],F_S[192],F_P[192],g,zeta_pow_alpha,gamma,target_predict_radiance"
                   << ", source=gpu"
                   << ", paired_descriptor_test=true"
                   << ", samples_per_volume=" << skewSampleCount
                   << "\n";

        const int volumeCount = dumpPairedDescriptorsAll ? static_cast<int>(dataList.size()) : std::min<int>(1, dataList.size());
        for (int volumeIndex = 0; volumeIndex < volumeCount; volumeIndex++) {
            const std::string& volumeName = dataList[volumeIndex];
            DumpPairedDescriptorsForVolume(relativePath + volumeName, volumeName, skewSampleCount, hostOutfile, gpuOutfile);
        }
        std::cout << "Wrote paired descriptor CSVs:\n"
                  << "  " << hostPath << "\n"
                  << "  " << gpuPath << "\n";
        return 0;
    }

    const int countAll = kSamplesPerModel * static_cast<int>(dataList.size());
    int computed = 0;

    //std::cout << dataName << std::endl;
    //std::cout << "countAll: " << countAll << std::endl;
    //return 0;

    auto start = std::chrono::steady_clock::now();

    std::ofstream outfile(dataPath + dataName);
    outfile << "# columns=" << kOutputColumns
            << ", layout=F_mu[192],F_S[192],F_P[192],g,zeta_pow_alpha,gamma,target_predict_radiance"
            << ", sampling=paper"
            << ", samples_per_model=" << kSamplesPerModel
            << ", condition_count_per_model=" << kConditionCountPerModel
            << ", zeta_range=[" << kMinZeta << "," << kMaxZeta << "]"
            << ", alpha=" << kAlbedoExponent
            << "\n";

    for (int dataIndex = 0; dataIndex < static_cast<int>(dataList.size()); dataIndex++) {
        printf("Processing %.2f%%\n", countAll > 0 ? 100.0f * computed / countAll : 0.0f);
        printf("Computing: %s\n", dataList[dataIndex].c_str());
        printf("Desired Size: %d\n", kSamplesPerModel);

        VolumeRender volume(relativePath + dataList[dataIndex]);
        std::vector<MRPNNDescriptorRow> rows;
        rows.reserve(kSamplesPerModel);

        for (int conditionIndex = 0; conditionIndex < kConditionCountPerModel; conditionIndex++) {
            const float3 lightDir = hash31sphere();
            const float epsilon = lerp(kEpsilonMin, kEpsilonMax, hash1());

            volume.Update_TR(lightDir, epsilon, true);

            printf("Condition %d/%d: epsilon=%.4f lightDir=(%.3f, %.3f, %.3f) count=%d\n",
                conditionIndex + 1,
                kConditionCountPerModel,
                epsilon,
                lightDir.x,
                lightDir.y,
                lightDir.z,
                kSamplesPerCondition);

            std::vector<MRPNNSamplePoint> samples;
            samples.reserve(kSamplesPerCondition);
            while (static_cast<int>(samples.size()) < kSamplesPerCondition) {
                const float g = lerp(0.0f, kMaxG, hash1());
                std::vector<MRPNNSamplePoint> oneSample;
                GetDesiredCountSample(volume, oneSample, 1, epsilon, g, lightDir);
                if (!oneSample.empty()) {
                    samples.push_back(oneSample.front());
                }
            }

            std::vector<float3> samplePositions;
            std::vector<float3> sampleDirs;
            std::vector<float3> sampleLightDirs;
            std::vector<float> sampleEpsilons;
            std::vector<float> sampleGs;
            std::vector<float> sampleScatters;

            samplePositions.reserve(samples.size());
            sampleDirs.reserve(samples.size());
            sampleLightDirs.reserve(samples.size());
            sampleEpsilons.reserve(samples.size());
            sampleGs.reserve(samples.size());
            sampleScatters.reserve(samples.size());

            for (const MRPNNSamplePoint& sample : samples) {
                samplePositions.push_back(sample.position);
                sampleDirs.push_back(sample.viewDir);
                sampleLightDirs.push_back(sample.lightDir);
                sampleEpsilons.push_back(sample.epsilon);
                sampleGs.push_back(sample.g);
                sampleScatters.push_back(sample.albedo);
            }

            // `GetSamples` is the same high-order target that `NNPredict` uses in its
            // commented reference path. Direct light is added later in `NNPredict`, so
            // this supervision already corresponds to PredictRadiance only.
            const float3 lightColor = {1.0f, 1.0f, 1.0f};
            const std::vector<float3> predictRadiances = volume.GetSamples(
                sampleEpsilons,
                samplePositions,
                sampleDirs,
                sampleLightDirs,
                sampleGs,
                sampleScatters,
                lightColor,
                kPathTraceScatterCount,
                kPathTraceSampleCount);

            for (size_t i = 0; i < samples.size(); i++) {
                volume.UpdateHGLut(samples[i].g);
                rows.push_back(BuildDescriptorRow(volume, samples[i], predictRadiances[i].x));
            }
            // AppendGpuDescriptorRows(volume, samples, predictRadiances, rows);
        }

        unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
        std::default_random_engine engine(seed);
        std::shuffle(rows.begin(), rows.end(), engine);

        for (size_t i = 0; i < rows.size(); i++) {
            if (i % std::max<size_t>(1, rows.size() / 8) == 0) {
                printf("Output Shuffle_Dataset: %.2f%%\n", 100.0f * static_cast<float>(i) / rows.size());
            }
            WriteRow(outfile, rows[i]);
            computed++;
        }
    }

    outfile.close();

    const auto end = std::chrono::steady_clock::now();
    const auto dtime = end - start;
    std::cout << "Render complete:\n";
    std::cout << "Time taken: "
              << std::chrono::duration_cast<std::chrono::hours>(dtime).count()
              << " hours\n";
    std::cout << "          : "
              << std::chrono::duration_cast<std::chrono::minutes>(dtime).count()
              << " minutes\n";
    std::cout << "          : "
              << std::chrono::duration_cast<std::chrono::seconds>(dtime).count()
              << " seconds\n";

    return 0;
}
