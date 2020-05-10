#pragma warning(push)
#pragma warning(disable: 4648)
#include <charconv>
#include <concepts>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

// Grumph!  These never should have been macros!
#undef min
#undef max
#pragma warning(pop)

template <typename T>
concept Enum = std::is_enum_v<T>;

template <Enum E>
using PrimitiveType = std::underlying_type_t<E>;

template <Enum E>
constexpr auto rep(E e) { return PrimitiveType<E>(e); }

template <Enum E>
constexpr E unit = { };


enum class RandomSeed : decltype(std::random_device{}()) { };

class RandomNumberGenerator
{
public:
    RandomNumberGenerator() = default;
    RandomNumberGenerator(RandomSeed seed):
        seed{ seed }, generator{ rep(seed) } { }

    template <typename T>
    auto generate(T&& distribution)
    {
        return std::forward<T>(distribution)(generator);
    }

    template <std::integral I>
    auto from_0_to_100()
    {
        std::uniform_int_distribution<I> dis{0, 100};
        return generate(dis);
    }

    template <std::floating_point F>
    auto from_0_to_1()
    {
        std::uniform_real_distribution<F> dis{ 0, 1 };
        return generate(dis);
    }

    auto initial_seed() const
    {
        return seed;
    }
private:
    RandomSeed seed = RandomSeed(std::random_device{}());
    std::mt19937 generator{ rep(seed) };
};

RandomNumberGenerator& random_generator()
{
    // The random number generator in <random> is HUGE and expensive to construct,
    // so we will only have one.
    static RandomNumberGenerator generator { RandomSeed(2608642933) };
    return generator;
}

template <std::floating_point F>
auto entropy(F disorder)
{
    return (random_generator().from_0_to_1<F>() * disorder) + F(1);
}

class PerlinNoiseGenerator
{
public:
    void seed_generator(std::vector<float> s)
    {
        this->seed = std::move(s);
    }

    void octave(int o)
    {
        octave_count = std::min(9, o);
    }

    int octave() const
    {
        return octave_count;
    }

    void scaling(float s)
    {
        scaling_bias = std::max(.2f, s);
    }

    float scaling() const
    {
        return scaling_bias;
    }

    std::vector<float> generate() const
    {
        int size = static_cast<int>(seed.size());
        std::vector<float> output(size);
        for (int x = 0; x < size; ++x)
        {
            float noise = .0f;
            float scale_acc = .0f;
            float scale = 1.f;

            for (int octave = 0; octave < octave_count; ++octave)
            {
                int pitch = (size >> octave);
                int sample1 = (x / pitch) * pitch;
                int sample2 = (sample1 + pitch) % size;

                float blend = static_cast<float>(x - sample1) / pitch;

                float sample = (1.f - blend) * seed[sample1] + blend * seed[sample2];

                scale_acc += scale;
                noise += sample * scale;
                scale = scale / scaling_bias;
            }
            output[x] = noise / scale_acc;
        }
        return output;
    }
private:
    std::vector<float> seed;
    float scaling_bias = .2f;
    int octave_count = 1;
};

enum class Width : int32_t { };
enum class Height : int32_t { };
enum class Radius : int32_t { };
enum class PixelWidth : int32_t { };
enum class PixelHeight : int32_t { };
struct ScreenInfo
{
    Width width;
    Height height;
    PixelWidth px_width;
    PixelHeight px_height;
};

enum class Row : int32_t { };
enum class Column : int32_t { };
class World
{
    enum class TransitionType
    {
        Blank,
        Grass,
        Crust,
        Rock,
        Lava,

        Count
    };

    struct Transition
    {
        int thickness = 0;
        TransitionType type = TransitionType::Blank;
    };

    using Columns = std::vector<olc::Pixel>;
    using Rows = std::vector<Columns>;

    struct ComputedTransition
    {
        Row start = unit<Row>;
        Height height = unit<Height>;
        TransitionType type = TransitionType::Blank;
    };
    using TransitionCollection = std::array<ComputedTransition, rep(TransitionType::Count)>;
public:
    static constexpr olc::Pixel Blank = olc::BLANK;
    static constexpr olc::Pixel Grass = olc::Pixel{ 51, 102, 0 };
    static constexpr olc::Pixel Crust = olc::Pixel{ 51, 25, 0 };
    static constexpr olc::Pixel Rock = olc::Pixel{ 64, 64, 64 };
    static constexpr olc::Pixel Lava = olc::RED;

    bool visible(const olc::vi2d& pos) const
    {
        return visible(Row(pos.y), Column(pos.x));
    }

    bool visible(Row row, Column col) const
    {
        if (!bounded(row, col))
        {
            return false;
        }
        return (*this)(row, col) != Blank;
    }

    bool bounded(const olc::vi2d& pos) const
    {
        return bounded(Row(pos.y), Column(pos.x));
    }

    bool bounded(Row row, Column col) const
    {
        if (rep(row) < 0)
        {
            return false;
        }
        
        if (rep(col) < 0)
        {
            return false;
        }

        return rep(row) < rep(height()) && rep(col) < rep(width());
    }

    olc::vi2d clamp(olc::vi2d pos) const
    {
        pos.x = std::min(rep(width()) - 1, pos.x);
        pos.y = std::min(rep(height()) - 1, pos.y);
        return pos;
    }

    Width width() const
    {
        return world_width;
    }

    Height height() const
    {
        return world_height;
    }

    const std::vector<olc::Pixel>& computed_world() const
    {
        return computed_pixels;
    }

    std::vector<olc::Pixel>& computed_world()
    {
        return computed_pixels;
    }

    void generate_world(Width width, Height height, const PerlinNoiseGenerator& noise_generator)
    {
        world_dimensions(width, height);

        world.resize(rep(width));
        std::vector<Transition> transition(rep(width));
        auto noise = noise_generator.generate();
        // Probability constants:
        const auto blank_to_grass = [&](Column col) { return rep(height) * noise[rep(col)]; };
        const auto grass_to_crust = rep(height) * .01f; // Take up a maximum of 1% of the screen
        const auto crust_to_rock = rep(height) * .05f;  // Take up a maximum of 5% of the screen
        const auto rock_to_lava  = rep(height) * .5f;   // Take up a maximum of 50% of the screen
        for (int i = 0; i != rep(width); ++i)
        {
            int h = static_cast<int>(entropy(.05f) * blank_to_grass(Column(i)));
            Transition t = { h, TransitionType::Blank };
            next_transition<TransitionType::Blank>(t, world[i]);

            h = static_cast<int>(entropy(.01f) * grass_to_crust);
            t = { h, TransitionType::Grass };
            next_transition<TransitionType::Grass>(t, world[i]);

            h = static_cast<int>(entropy(.02f) * crust_to_rock);
            t = { h, TransitionType::Crust };
            next_transition<TransitionType::Crust>(t, world[i]);

            h = static_cast<int>(entropy(.02f) * rock_to_lava);
            t = { h, TransitionType::Rock };
            next_transition<TransitionType::Rock>(t, world[i]);

            auto row_start = Row(rep(world[i][rep(TransitionType::Rock)].start) + rep(world[i][rep(TransitionType::Rock)].height));
            world[i][rep(TransitionType::Lava)] = { row_start, Height(rep(height) - rep(row_start)), TransitionType::Lava };
        }
        precompute_world(width, height);
    }

    const olc::Pixel& operator()(const olc::vi2d& pos) const
    {
        return (*this)(Row(pos.y), Column(pos.x));
    }

    const olc::Pixel& operator()(Row row, Column col) const
    {
        return computed_world()[rep(col) + rep(row) * world.size()];
    }

    olc::Pixel& operator()(const olc::vi2d& pos)
    {
        return (*this)(Row(pos.y), Column(pos.x));
    }

    olc::Pixel& operator()(Row row, Column col)
    {
        return computed_world()[rep(col) + rep(row) * world.size()];
    }

    olc::vf2d normal_to(const olc::vi2d& pos) const
    {
        return normal_to(Row(pos.y), Column(pos.x));
    }

    olc::vf2d normal_to(Row row, Column col) const
    {
        constexpr int radius = 3;
        float avg_x = 0.f;
        float avg_y = 0.f;
        for (int x = -radius + rep(col), last_x = rep(col) + radius; x <= last_x; ++x)
        {
            for (int y = -radius + rep(row), last_y = radius + rep(row); y <= last_y; ++y)
            {
                if (visible(Row(y), Column(x)))
                {
                    avg_x -= x - rep(col);
                    avg_y -= y - rep(row);
                }
            }
        }
        float len = std::sqrt(avg_x * avg_x + avg_y * avg_y);
        return { avg_x / len, avg_y / len };
    }

    // Get the number of visible pixels at a given point with a certain radius (implemented as a square area).
    int sample_visible(Row row, Column col, Width width)
    {
        int count = 0;
        for (int x = -rep(width) + rep(col), last_x = rep(width) + rep(col); x <= last_x; ++x)
        {
            for (int y = -rep(width) + rep(row), last_y = rep(width) + rep(row); y <= last_y; ++y)
            {
                if (visible(Row(y), Column(x)))
                {
                    ++count;
                }
            }
        }
        return count;
    }
private:
    void world_dimensions(Width width, Height height)
    {
        world_width = width;
        world_height = height;
    }

    // Only used for fetching pixel values internally in precompute_world.
    const olc::Pixel& pixel_for_compute(Row row, Column col) const
    {
        const auto& c = world[rep(col)];
        for (const auto& t : c)
        {
            if (rep(row) <= (rep(t.start) + rep(t.height)))
            {
                switch (t.type)
                {
                case TransitionType::Blank: return Blank;
                case TransitionType::Grass: return Grass;
                case TransitionType::Crust: return Crust;
                case TransitionType::Rock: return Rock;
                case TransitionType::Lava: return Lava;
                }
            }
        }
        return Blank;
    }

    template <TransitionType T>
    void update(Transition& t, TransitionCollection& collection)
    {
        ++t.thickness;
        if constexpr (T == TransitionType::Blank)
        {
            collection[rep(T)] = { Row(0), Height(t.thickness), T };
        }
        else
        {
            constexpr TransitionType prev_type = TransitionType(rep(T) - 1);
            const auto& prev = collection[rep(prev_type)];
            auto previous_row_end = Row(rep(prev.start) + rep(prev.height));
            collection[rep(T)] = { previous_row_end, Height(t.thickness), T };
        }
    }

    template <TransitionType From>
    void next_transition(Transition& t, TransitionCollection& collection)
    {
        update<From>(t, collection);
        constexpr auto next = TransitionType(rep(From) + 1);
        t = Transition{ 0, next };
    }

    void precompute_world(Width width, Height height)
    {
        computed_pixels.resize(rep(width) * rep(height));
        for (int i = 0; i != rep(height); ++i)
        {
            for (int j = 0; j != rep(width); ++j)
            {
                computed_pixels[j + i * rep(width)] = pixel_for_compute(Row(i), Column(j));
            }
        }
    }

    std::vector<TransitionCollection> world;
    std::vector<olc::Pixel> computed_pixels;
    Width world_width = Width(0);
    Height world_height = Height(0);
};

struct RayCastResult
{
    olc::vi2d previous_point;
    olc::vi2d impact_point;
};

std::optional<RayCastResult> cast_ray(const World& world, const olc::vi2d& start, const olc::vi2d& end)
{
    // see https://www.gamedev.net/reference/articles/article1275.asp.
    auto delta = end - start;
    delta.x = std::abs(delta.x);
    delta.y = std::abs(delta.y);
    olc::vi2d increment1;
    olc::vi2d increment2;
    if (end.x >= start.x)
    {
        increment1.x = 1;
        increment2.x = 1;
    }
    else
    {
        increment1.x = -1;
        increment2.x = -1;
    }

    if (end.y >= start.y)
    {
        increment1.y = 1;
        increment2.y = 1;
    }
    else
    {
        increment1.y = -1;
        increment2.y = -1;
    }

    int den = 0;
    int num = 0;
    int num_add = 0;
    int num_pixels = 0;
    if (delta.x >= delta.y)
    {
        increment1.x = 0;
        increment2.y = 0;
        den = delta.x;
        num = delta.x / 2;
        num_add = delta.y;
        num_pixels = delta.x;
    }
    else
    {
        increment2.x = 0;
        increment1.y = 0;
        den = delta.y;
        num = delta.y / 2;
        num_add = delta.x;
        num_pixels = delta.y;
    }
    auto current = start;
    auto prev = start;

    for (int pixel = 0; pixel <= num_pixels; ++pixel)
    {
        if (world.visible(current))
        {
            return { { prev, current } };
        }

        prev = current;

        num += num_add;

        if (num >= den)
        {
            num -= den;
            current += increment1;
        }

        current += increment2;
    }
    return { };
}

struct PhysicalProperties
{
    float stickyness = 150.f;
    float friction = .85f;
};

class PhysicsPixel
{
public:
    PhysicsPixel(const olc::vi2d& pos, const olc::vf2d& velocity, olc::Pixel color, Radius radius, PhysicalProperties properties = { }):
        pos{ pos }, vel{ velocity }, pixel_color{ color }, radius{ radius }, properties{ properties } { }

    void execute_step(World* world)
    {
        if (auto collision = cast_ray(*world, prev_pos, pos))
        {
            collide(*collision, world);
        }

        prev_pos = pos;
        if (!world->bounded(pos))
        {
            if (pos.y >= rep(world->height()))
            {
                adhere_to(world, world->clamp(pos));
            }
            else
            {
                exploded = true;
            }
        }
    }

    bool dead() const
    {
        return exploded;
    }

    const auto& position() const
    {
        return pos;
    }

    auto& position()
    {
        return pos;
    }

    const auto& velocity() const
    {
        return vel;
    }

    auto& velocity()
    {
        return vel;
    }

    auto& color() const
    {
        return pixel_color;
    }
private:
    void collide(const RayCastResult& result, World* world)
    {
        if (can_stick())
        {
            adhere_to(world, result.previous_point);
            return;
        }

        auto normal = world->normal_to(result.impact_point);
        float d = 2 * normal.dot(velocity());
        velocity() -= normal * d;
        velocity() *= properties.friction;
        pos = result.previous_point;
    }

    void adhere_to(World* world, const olc::vi2d& point)
    {
        (*world)(point) = color();
        exploded = true;
    }

    bool can_stick() const
    {
        return velocity().mag2() < (properties.stickyness * properties.stickyness);
    }

    olc::Pixel pixel_color;
    olc::vi2d pos;
    olc::vf2d vel;
    Radius radius;
    olc::vi2d prev_pos = pos;
    PhysicalProperties properties;
    bool exploded = false;
};

class PhysicsEngine
{
    static constexpr int d_time = 16;
    static constexpr float d_time_s = static_cast<float>(d_time) / 1000.f;
public:
    void add(const PhysicsPixel& pixel)
    {
        pixels_objects.push_back(pixel);
    }

    void update(float elapsed_time, World* world)
    {
        // Compute steps that can fit into this delta.
        int steps = static_cast<int>((elapsed_time + static_cast<float>(left_over_time)) / d_time_s);
        steps = std::min(steps, 1);
        left_over_time = static_cast<int>(d_time_s) - (steps * d_time);

        constexpr int cull_dead_threshold = 50;
        int dead_count = 0;
        for (int i = 1; i <= steps; ++i)
        {
            for (auto& pixel : pixels_objects)
            {
                if (pixel.dead())
                {
                    ++dead_count;
                    continue;
                }

                // Add gravity.
                pixel.velocity().y += 980.f * d_time_s;

                // Always add x velocity.
                pixel.position() += pixel.velocity() * d_time_s;

                pixel.execute_step(world);
            }
        }

        if (dead_count >= cull_dead_threshold)
        {
            pixels_objects.erase(std::remove_if(begin(pixels_objects),
                                        end(pixels_objects),
                                        [](const PhysicsPixel& pixel)
                                        {
                                            return pixel.dead();
                                        }),
                                    end(pixels_objects));
        }
    }

    const auto& pixels() const
    {
        return pixels_objects;
    }
private:
    int left_over_time = 0;
    std::vector<PhysicsPixel> pixels_objects;
};

void explode(World* world, PhysicsEngine* physics_engine, const olc::vi2d& pos, float radius)
{
    float r_squared = radius * radius;
    int x = static_cast<int>(pos.x - radius);
    int x_end = static_cast<int>(pos.x + radius);
    for (;x != x_end; ++x)
    {
        int y = static_cast<int>(pos.y - radius);
        int y_end = static_cast<int>(pos.y + radius);
        for (; y != y_end; ++y)
        {
            if (world->bounded(Row(y), Column(x)))
            {
                float x_diff = static_cast<float>(x - pos.x);
                float y_diff = static_cast<float>(y - pos.y);
                float distance = x_diff * x_diff + y_diff * y_diff;
                if (distance < r_squared)
                {
                    (*world)(Row(y), Column(x)) = World::Blank;
                }
            }
        }
    }

    // Turn all of the solid pixels in the span of the diameter above the destroyed terrain into physics
    // pixels to animate them.
    constexpr PhysicalProperties destroyed_props = { .stickyness = 1500.f };
    for (x = static_cast<int>(pos.x - radius); x != x_end; ++x)
    {
        for (int y = pos.y; y > 0; --y)
        {
            if (world->visible(Row(y), Column(x)))
            {
                physics_engine->add(PhysicsPixel{ { x, y },
                                                  { 0.f, 0.f },
                                                  (*world)(Row(y), Column(x)), Radius(1),
                                                  destroyed_props });
                (*world)(Row(y), Column(x)) = World::Blank;
            }
        }
    }
}

class Game : olc::PixelGameEngine
{
    using Base = olc::PixelGameEngine;
public:
    Game()
    {
        sAppName = "Game";
    }

    bool OnUserCreate() override
    {
        return true;
    }

    bool OnUserUpdate(float elapsed_time) override
    {
        bool changed = false;
        if (GetKey(olc::Key::RIGHT).bReleased)
        {
            noise_generator.octave(noise_generator.octave() + 1);
            changed = true;
        }

        if (GetKey(olc::Key::LEFT).bReleased)
        {
            noise_generator.octave(noise_generator.octave() - 1);
            changed = true;
        }

        if (GetKey(olc::Key::UP).bReleased)
        {
            noise_generator.scaling(noise_generator.scaling() + .2f);
            changed = true;
        }

        if (GetKey(olc::Key::DOWN).bReleased)
        {
            noise_generator.scaling(noise_generator.scaling() - .2f);
            changed = true;
        }

        if (GetKey(olc::Key::R).bReleased)
        {
            noise_generator.seed_generator(noise_seed(rep(screen_info.width)));
            changed = true;
        }

        if (changed)
        {
            Clear(olc::BLACK);
            world.generate_world(screen_info.width, screen_info.height, noise_generator);
        }

        physics_engine.update(elapsed_time, &world);

        if (GetMouse(1).bHeld)
        {
            std::uniform_real_distribution<float> dis_velocity_x{ 50.f, 1500.f };
            std::uniform_real_distribution<float> dis_velocity_y{ 50.f, 500.f };
            std::uniform_int_distribution<int> dis_color{ 0, 255 };
            for (int i = 0; i != 10; ++i)
            {
                olc::vf2d velocity{ random_generator().generate(dis_velocity_x), random_generator().generate(dis_velocity_y) };
                olc::Pixel color {
                    static_cast<uint8_t>(random_generator().generate(dis_color)),
                    static_cast<uint8_t>(random_generator().generate(dis_color)),
                    static_cast<uint8_t>(random_generator().generate(dis_color)) };
                physics_engine.add(PhysicsPixel{ { GetMouseX(), GetMouseY() }, velocity, color, Radius(1) });
            }
        }

        if (GetMouse(0).bPressed)
        {
            explode(&world, &physics_engine, { GetMouseX(), GetMouseY() }, entropy(1.f) * 20.f);
            changed = true;
        }

        if (GetKey(olc::Key::SPACE).bHeld)
        {
            for (int i = 0; i != rep(world.height()); ++i)
            {
                for (int j = 0; j != rep(world.width()); ++j)
                {
                    if (world.visible(Row(i), Column(j)))
                    {
                        Draw(j, i, world(Row(i), Column(j)));
                    }
                }
            }
        }
        else
        {
            std::copy(begin(world.computed_world()), end(world.computed_world()), GetDrawTarget()->GetData());
        }

        if (GetKey(olc::Key::N).bReleased)
        {
            if (!computed_normals.empty())
            {
                computed_normals.clear();
            }
            else
            {
                compute_normals();
            }
        }

        if (!computed_normals.empty())
        {
            if (changed)
            {
                compute_normals();
            }
            for (const auto& [start, end] : computed_normals)
            {
                DrawLine(start, end, olc::RED);
            }
        }

        draw_physics();

        if (changed)
        {
            std::stringstream ss;
            ss << "Octaves: " << noise_generator.octave() << " Scaling: " << noise_generator.scaling();
            display_text = ss.str();
        }
        DrawString({ 10, 10 }, display_text);

        return !GetKey(olc::Key::ESCAPE).bPressed;
    }

    auto Construct(ScreenInfo info)
    {
        screen_info = info;
        noise_generator.seed_generator(noise_seed(rep(info.width)));
        world.generate_world(info.width, info.height, noise_generator);
        return Base::Construct(rep(info.width), rep(info.height), rep(info.px_width), rep(info.px_height), false, true);
    }

    // Inject base methods...
    using Base::Start;

private:
    void draw_physics()
    {
        int alive_particles = 0;
        for (const auto& pixel : physics_engine.pixels())
        {
            if (!pixel.dead())
            {
                ++alive_particles;
                Draw(pixel.position(), pixel.color());
            }
        }
        {
            char buf[] = "Particle Count: xxxxxxxxxxxx";
            auto [p, ec] = std::to_chars(buf + sizeof("Particle Count:"), std::end(buf), alive_particles);
            *p = '\0';
            DrawString({ 10, 20 }, {buf, p });
        }
    }

    void compute_normals()
    {
        computed_normals.clear();
        constexpr int increment = 5;
        for (int x = 0; x < rep(world.width()); x += increment)
        {
            for (int y = 0; y < rep(world.height()); y += increment)
            {
                int sample = world.sample_visible(Row(y), Column(x), Width(5));
                if (sample < 110 && sample > 30)
                {
                    olc::vf2d normal = world.normal_to(Row(y), Column(x));
                    if (normal.mag2() > 0.f)
                    {
                        computed_normals.push_back({ { x, y },
                                                    { x + static_cast<int>(10 * normal.x), y + static_cast<int>(10 * normal.y) } });
                    }
                }
            }
        }
    }
    std::vector<float> noise_seed(int size) const
    {
        std::vector<float> seed(size);
        for (float& f : seed)
        {
            f = random_generator().from_0_to_1<float>();
        }
        return seed;
    }

    ScreenInfo screen_info;
    World world;
    PerlinNoiseGenerator noise_generator;
    PhysicsEngine physics_engine;

    // Debug info stuff
    std::vector<std::pair<olc::vi2d, olc::vi2d>> computed_normals;
    std::string display_text;
};

int main()
{
    Game game;
    if (game.Construct({ Width(1024), Height(768), PixelWidth(1), PixelHeight(1) }))
    {
        game.Start();
    }
}