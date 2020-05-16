#pragma warning(push)
#pragma warning(disable: 4648)
#include <cassert>

#include <charconv>
#include <concepts>
#include <execution>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

// Grumph!  These never should have been macros!
#undef min
#undef max
#pragma warning(pop)

// Resources used:
// Primary idea (pixel terrain):
// * https://gamedevelopment.tutsplus.com/tutorials/coding-destructible-pixel-terrain-how-to-make-everything-explode--gamedev-45
//
// Collisions:
// * https://spicyyoghurt.com/tutorials/html5-javascript-game-development/collision-detection-physics
// * https://gamedevelopment.tutsplus.com/tutorials/when-worlds-collide-simulating-circle-circle-collisions--gamedev-769
//
// Line Drawing:
// * https://www.gamedev.net/reference/articles/article1275.asp
//
// QuadTree:
// * https://gamedevelopment.tutsplus.com/tutorials/quick-tip-use-quadtrees-to-detect-likely-collisions-in-2d-space--gamedev-374
//
// % Chance:
// * https://gamedev.stackexchange.com/questions/22891/designing-drops-system-how-and-where-chance-of-drops-are-defined

template <typename T>
concept Enum = std::is_enum_v<T>;

template <Enum E>
using PrimitiveType = std::underlying_type_t<E>;

template <Enum E>
constexpr auto rep(E e) { return PrimitiveType<E>(e); }

template <Enum E>
constexpr auto retract(E e) { return E(rep(e) - 1); }

template <Enum E>
constexpr auto extend(E e) { return E(rep(e) + 1); }

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
    static RandomNumberGenerator generator;
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
                // No future octave will solve this.
                if (pitch == 0)
                {
                    break;
                }
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

std::vector<float> noise_seed(int size)
{
    std::vector<float> seed(size);
    for (float& f : seed)
    {
        f = random_generator().from_0_to_1<float>();
    }
    return seed;
}

template <typename T>
std::vector<float> noise_seed(int size, T&& distribution)
{
    std::vector<float> seed(size);
    for (float& f : seed)
    {
        f = random_generator().generate(std::forward<T>(distribution));
    }
    return seed;
}

enum class Width : int32_t { };
enum class Height : int32_t { };
enum class Radius : int32_t { };
enum class Weight : int32_t { };
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

enum class TransitionType
{
    Mountains,
    Valley,
    Lowlands,
    Beach,

    Count
};

template <typename T>
concept Countable = requires(T) {
    T::Count;
};

template <typename T>
concept TransitionElementsModel = requires(T) {
    T::Blank;
} && Countable<T>;

template <Countable T>
constexpr bool last_of(T t)
{
    return extend(t) == T::Count;
}

template <TransitionElementsModel M, TransitionType Type>
struct ComputedTransition
{
    struct Data
    {
        Row start = unit<Row>;
        Height height = unit<Height>;
        M type = M::Blank;
    };

    static constexpr TransitionType type = Type;

    Data collection[rep(M::Count)] = { };
};

template <TransitionType type>
struct TransitionProperties;

namespace Colors
{
    constexpr auto blank = olc::BLANK;
    constexpr auto snow = olc::WHITE;
    constexpr auto crust = olc::Pixel{ 51, 25, 0 };
    constexpr auto rock = olc::Pixel{ 64, 64, 64 };
    constexpr auto lava = olc::RED;
    constexpr auto grass = olc::Pixel{ 51, 102, 0 };
    constexpr auto clay = olc::Pixel{ 95, 25, 0 };
    constexpr auto sand = olc::YELLOW;
}

template <>
struct TransitionProperties<TransitionType::Mountains>
{
    enum class TransitionElements
    {
        Blank,
        Snow,
        Crust,
        Rock,
        Lava,

        Count
    };

    static constexpr olc::Pixel transition_colors[rep(TransitionElements::Count)] =
        {
            Colors::blank,
            Colors::snow,
            Colors::crust,
            Colors::rock,
            Colors::lava,
        };
};

template <>
struct TransitionProperties<TransitionType::Valley>
{
    enum class TransitionElements
    {
        Blank,
        Grass,
        Crust,
        Clay,
        Rock,

        Count
    };

    static constexpr olc::Pixel transition_colors[rep(TransitionElements::Count)] =
        {
            Colors::blank,
            Colors::grass,
            Colors::crust,
            Colors::clay,
            Colors::rock,
        };
};

template <>
struct TransitionProperties<TransitionType::Lowlands>
{
    enum class TransitionElements
    {
        Blank,
        Grass,
        Crust,
        Clay,
        Rock,

        Count
    };

    static constexpr olc::Pixel transition_colors[rep(TransitionElements::Count)] =
        {
            Colors::blank,
            Colors::grass,
            Colors::crust,
            Colors::clay,
            Colors::rock,
        };
};

template <>
struct TransitionProperties<TransitionType::Beach>
{
    enum class TransitionElements
    {
        Blank,
        Sand,
        Crust,
        Clay,
        Rock,

        Count
    };

    static constexpr olc::Pixel transition_colors[rep(TransitionElements::Count)] =
        {
            Colors::blank,
            Colors::sand,
            Colors::crust,
            Colors::clay,
            Colors::rock,
        };
};

template <TransitionType T>
using ElementsOf = typename TransitionProperties<T>::TransitionElements;

template <TransitionType T>
using ComputedTransitionFor = ComputedTransition<ElementsOf<T>, T>;

// Ugh... I wish pack expansions were easier...
template <typename>
struct TransitionVariantImpl;

template <std::size_t... Transitions>
struct TransitionVariantImpl<std::index_sequence<Transitions...>>
{
    using Type = std::variant<ComputedTransitionFor<TransitionType(Transitions)>...>;
};

using TransitionVariant = TransitionVariantImpl<std::make_index_sequence<rep(TransitionType::Count)>>::Type;

constexpr TransitionType type_of(const TransitionVariant& v)
{
    return TransitionType(v.index());
}

constexpr TransitionVariant transition_variant(TransitionType type)
{
    switch (type)
    {
    case TransitionType::Mountains:
        return ComputedTransitionFor<TransitionType::Mountains>{ };
    case TransitionType::Valley:
        return ComputedTransitionFor<TransitionType::Valley>{ };
    case TransitionType::Lowlands:
        return ComputedTransitionFor<TransitionType::Lowlands>{ };
    case TransitionType::Beach:
        return ComputedTransitionFor<TransitionType::Beach>{ };
    }
    return { };
}

class WorldHeightGenerator
{
public:
    void generate(Width width, Height height, PerlinNoiseGenerator* noise_generator)
    {
        world.resize(rep(width));
        generate_terrain_types(width);
        generate_height_maps(width, height, noise_generator);
    }

    olc::Pixel pixel_for_compute(Row row, Column col) const
    {
        const auto& data = world[rep(col)];
        switch (type_of(data))
        {
        case TransitionType::Mountains:
            return pixel_for_compute_internal(as_collection<TransitionType::Mountains>(col), row);
        case TransitionType::Valley:
            return pixel_for_compute_internal(as_collection<TransitionType::Valley>(col), row);
        case TransitionType::Lowlands:
            return pixel_for_compute_internal(as_collection<TransitionType::Lowlands>(col), row);
        case TransitionType::Beach:
            return pixel_for_compute_internal(as_collection<TransitionType::Beach>(col), row);
        }
        return olc::BLANK;
    }

private:
    template <typename T>
    olc::Pixel pixel_for_compute_internal(const T& data, Row row) const
    {
        for (const auto& transition : data.collection)
        {
            if (rep(row) <= (rep(transition.start) + rep(transition.height)))
            {
                return TransitionProperties<T::type>::transition_colors[rep(transition.type)];
            }
        }
        return olc::BLANK;
    }

    static TransitionType initial_state()
    {
        std::uniform_int_distribution<int> dis{ rep(unit<TransitionType>), rep(TransitionType::Count) };
        return TransitionType(random_generator().generate(dis));
    }

    // Terrain type generator.
    void generate_terrain_types(Width width)
    {
        // Populate the initial state.
        transition_to(initial_state(), Column(0));
        for (int i = 1; i != rep(width); ++i)
        {
            terrain_handle_state(Column(i));
        }
    }

    // State transition handlers for terrain types.
    void terrain_handle_state(Column col)
    {
        Column previous = retract(col);
        switch (type_of(world[rep(previous)]))
        {
        case TransitionType::Mountains:
            terrain_handle_mountain_state(col);
            break;
        case TransitionType::Valley:
            terrain_handle_valley_state(col);
            break;
        case TransitionType::Lowlands:
            terrain_handle_lowlands_state(col);
            break;
        case TransitionType::Beach:
            terrain_handle_beach_state(col);
            break;
        }
    }

    void terrain_handle_mountain_state(Column col)
    {
        const float max_width = static_cast<float>(world.size()) * .7f;
        const float min_width = static_cast<float>(world.size()) * .1f;
        constexpr int chance_for_next = 0;
        if (current_state_width < static_cast<int>(min_width))
        {
            continue_transition(col);
            return;
        }

        if (current_state_width >= static_cast<int>(max_width))
        {
            transition_to(extend(TransitionType::Mountains), col);
            return;
        }
        // Roll the dice!
        if (random_generator().from_0_to_100<int>() < chance_for_next)
        {
            transition_to(extend(TransitionType::Mountains), col);
        }
        else
        {
            continue_transition(col);
        }
    }

    void terrain_handle_valley_state(Column col)
    {
        const float max_width = static_cast<float>(world.size()) * .2f;
        const float min_width = static_cast<float>(world.size()) * .1f;
        constexpr int chance_for_next = 1;
        if (current_state_width < static_cast<int>(min_width))
        {
            continue_transition(col);
            return;
        }

        if (current_state_width >= static_cast<int>(max_width))
        {
            if (random_generator().from_0_to_100<int>() < chance_for_next)
            {
                transition_to(extend(TransitionType::Valley), col);
            }
            else
            {
                transition_to(retract(TransitionType::Valley), col);
            }
            return;
        }
        // Roll the dice!
        chance_transition<TransitionType::Valley>(col);
    }

    void terrain_handle_lowlands_state(Column col)
    {
        const float max_width = static_cast<float>(world.size()) * .4f;
        const float min_width = static_cast<float>(world.size()) * .2f;
        constexpr int chance_for_next = 1;
        if (current_state_width < static_cast<int>(min_width))
        {
            continue_transition(col);
            return;
        }

        if (current_state_width >= static_cast<int>(max_width))
        {
            if (random_generator().from_0_to_100<int>() < chance_for_next)
            {
                transition_to(extend(TransitionType::Lowlands), col);
            }
            else
            {
                transition_to(retract(TransitionType::Lowlands), col);
            }
            return;
        }
        // Roll the dice!
        chance_transition<TransitionType::Lowlands>(col);
    }

    void terrain_handle_beach_state(Column col)
    {
        const float max_width = static_cast<float>(world.size()) * .2f;
        const float min_width = static_cast<float>(world.size()) * .08f;
        constexpr int chance_for_previous = 1;
        if (current_state_width < static_cast<int>(min_width))
        {
            continue_transition(col);
            return;
        }

        if (current_state_width >= static_cast<int>(max_width))
        {
            transition_to(retract(TransitionType::Beach), col);
            return;
        }
        // Roll the dice!
        if (random_generator().from_0_to_100<int>() < chance_for_previous)
        {
            transition_to(retract(TransitionType::Beach), col);
        }
        else
        {
            continue_transition(col);
        }
        // Add more if we hit this.
        static_assert(last_of(TransitionType::Beach));
    }

    void continue_transition(Column col)
    {
        world[rep(col)] = world[rep(retract(col))];
        ++current_state_width;
    }

    void transition_to(TransitionType type, Column col)
    {
        world[rep(col)] = transition_variant(type);
        current_state_width = 1;
    }

    template <TransitionType Type>
    void chance_transition(Column col)
    {
        static_assert(!last_of(Type));
        static_assert(Type != unit<TransitionType>);
        std::uniform_int_distribution<int> dis{ rep(retract(Type)), rep(extend(Type)) };
        auto new_type = TransitionType(random_generator().generate(dis));
        if (new_type == Type)
        {
            continue_transition(col);
        }
        else
        {
            transition_to(new_type, col);
        }
    }

    // Height map generation.
    void generate_height_maps(Width width, Height height, PerlinNoiseGenerator* noise_generator)
    {
        for (int i = 0; i < rep(width);)
        {
            i += height_map_handle_state(Column(i), height, noise_generator);
        }
    }

    // Returns the new column to be processed by caller.
    int height_map_handle_state(Column col, Height height, PerlinNoiseGenerator* noise_generator)
    {
        switch (type_of(world[rep(col)]))
        {
        case TransitionType::Mountains:
            return height_map_handle_mountain_state(col, height, noise_generator);
        case TransitionType::Valley:
            return height_map_handle_valley_state(col, height, noise_generator);
        case TransitionType::Lowlands:
            return height_map_handle_lowlands_state(col, height, noise_generator);
        case TransitionType::Beach:
            return height_map_handle_beach_state(col, height, noise_generator);
        }
        return 0;
    }

    int height_map_handle_mountain_state(Column col, Height height, PerlinNoiseGenerator* noise_generator)
    {
        // Find the width of this terrain.
        constexpr auto T = TransitionType::Mountains;
        int length = terrain_length_from<T>(col);
        std::uniform_real_distribution<float> dis{ 0.05f, 1.f };
        noise_generator->seed_generator(transitional_noise(col, height, Width(length), dis));
        noise_generator->octave(8);
        noise_generator->scaling(1.2f);
        auto noise = noise_generator->generate();
        // Probability constants:
        const auto blank_to_snow = [&](Column off) { return rep(height) * noise[rep(off) - rep(col)]; };
        const auto snow_to_crust = rep(height) * .01f;
        const auto crust_to_rock = rep(height) * .05f;
        const auto rock_to_lava = rep(height) * .5f;
        for (int i = rep(col); i != rep(col) + length; ++i)
        {
            using E = ElementsOf<T>;
            int h = static_cast<int>(entropy(.05f) * blank_to_snow(Column(i)));
            populate_transition<T, E::Blank>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * snow_to_crust);
            populate_transition<T, E::Snow>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * crust_to_rock);
            populate_transition<T, E::Crust>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * rock_to_lava);
            populate_transition<T, E::Rock>(as_collection<T>(Column(i)), Height(h));

            // The last one we simply pass the height of the map and it figures out the rest.
            populate_transition<T, E::Lava>(as_collection<T>(Column(i)), height);
            static_assert(last_of(E::Lava));
        }
        return length;
    }

    int height_map_handle_valley_state(Column col, Height height, PerlinNoiseGenerator* noise_generator)
    {
        // Find the width of this terrain.
        constexpr auto T = TransitionType::Valley;
        int length = terrain_length_from<T>(col);
        std::uniform_real_distribution<float> dis{ 0.2f, 0.3f };
        noise_generator->seed_generator(transitional_noise(col, height, Width(length), dis));
        noise_generator->octave(4);
        noise_generator->scaling(1.2f);
        auto noise = noise_generator->generate();
        // Probability constants:
        const auto blank_to_grass = [&](Column off) { return rep(height) * noise[rep(off) - rep(col)]; };
        const auto grass_to_crust = rep(height) * .01f;
        const auto crust_to_clay = rep(height) * .05f;
        const auto clay_to_rock = rep(height) * .5f;
        for (int i = rep(col); i != rep(col) + length; ++i)
        {
            using E = ElementsOf<T>;
            int h = static_cast<int>(entropy(.03f) * blank_to_grass(Column(i)));
            populate_transition<T, E::Blank>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * grass_to_crust);
            populate_transition<T, E::Grass>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * crust_to_clay);
            populate_transition<T, E::Crust>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * clay_to_rock);
            populate_transition<T, E::Clay>(as_collection<T>(Column(i)), Height(h));

            // The last one we simply pass the height of the map and it figures out the rest.
            populate_transition<T, E::Rock>(as_collection<T>(Column(i)), height);
            static_assert(last_of(E::Rock));
        }
        return length;
    }

    int height_map_handle_lowlands_state(Column col, Height height, PerlinNoiseGenerator* noise_generator)
    {
        // Find the width of this terrain.
        constexpr auto T = TransitionType::Lowlands;
        int length = terrain_length_from<T>(col);
        std::uniform_real_distribution<float> dis{ 0.8f, 0.9f };
        noise_generator->seed_generator(transitional_noise(col, height, Width(length), dis));
        noise_generator->octave(2);
        noise_generator->scaling(1.2f);
        auto noise = noise_generator->generate();
        // Probability constants:
        const auto blank_to_grass = [&](Column off) { return rep(height) * noise[rep(off) - rep(col)]; };
        const auto grass_to_crust = rep(height) * .01f;
        const auto crust_to_clay = rep(height) * .05f;
        const auto clay_to_rock = rep(height) * .5f;
        for (int i = rep(col); i != rep(col) + length; ++i)
        {
            using E = ElementsOf<T>;
            int h = static_cast<int>(entropy(.01f) * blank_to_grass(Column(i)));
            populate_transition<T, E::Blank>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * grass_to_crust);
            populate_transition<T, E::Grass>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * crust_to_clay);
            populate_transition<T, E::Crust>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * clay_to_rock);
            populate_transition<T, E::Clay>(as_collection<T>(Column(i)), Height(h));

            // The last one we simply pass the height of the map and it figures out the rest.
            populate_transition<T, E::Rock>(as_collection<T>(Column(i)), height);
            static_assert(last_of(E::Rock));
        }
        return length;
    }

    int height_map_handle_beach_state(Column col, Height height, PerlinNoiseGenerator* noise_generator)
    {
        // Find the width of this terrain.
        constexpr auto T = TransitionType::Beach;
        int length = terrain_length_from<T>(col);
        std::uniform_real_distribution<float> dis{ 0.95f, 1.f };
        noise_generator->seed_generator(transitional_noise(col, height, Width(length), dis));
        noise_generator->octave(1);
        noise_generator->scaling(2.2f);
        auto noise = noise_generator->generate();
        // Probability constants:
        const auto blank_to_sand = [&](Column off) { return rep(height) * noise[rep(off) - rep(col)]; };
        const auto sand_to_crust = rep(height) * .01f;
        const auto crust_to_clay = rep(height) * .05f;
        const auto clay_to_rock = rep(height) * .5f;
        for (int i = rep(col); i != rep(col) + length; ++i)
        {
            using E = ElementsOf<T>;
            int h = static_cast<int>(blank_to_sand(Column(i)));
            populate_transition<T, E::Blank>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * sand_to_crust);
            populate_transition<T, E::Sand>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * crust_to_clay);
            populate_transition<T, E::Crust>(as_collection<T>(Column(i)), Height(h));

            h = static_cast<int>(entropy(.05f) * clay_to_rock);
            populate_transition<T, E::Clay>(as_collection<T>(Column(i)), Height(h));

            // The last one we simply pass the height of the map and it figures out the rest.
            populate_transition<T, E::Rock>(as_collection<T>(Column(i)), height);
            static_assert(last_of(E::Rock));
        }
        return length;
    }

    template <TransitionType Type>
    ComputedTransitionFor<Type>& as_collection(Column col)
    {
        return std::get<ComputedTransitionFor<Type>>(world[rep(col)]);
    }

    template <TransitionType Type>
    const ComputedTransitionFor<Type>& as_collection(Column col) const
    {
        return std::get<ComputedTransitionFor<Type>>(world[rep(col)]);
    }

    template <TransitionType Type, ElementsOf<Type> E, typename Collection>
    void populate_transition(Collection& collection, Height height)
    {
        // First height.
        if constexpr (E == ElementsOf<Type>::Blank)
        {
            collection.collection[rep(E)] = { Row(0), height, E };
        }
        // Last height (goes to bottom).
        else if constexpr (last_of(E))
        {
            const auto& prev = collection.collection[rep(retract(E))];
            auto prev_row_end = Row(rep(prev.start) + rep(prev.height));
            collection.collection[rep(E)] = { prev_row_end, Height(rep(height) - rep(prev_row_end)), E };
        }
        else
        {
            const auto& prev = collection.collection[rep(retract(E))];
            auto prev_row_end = Row(rep(prev.start) + rep(prev.height));
            collection.collection[rep(E)] = { prev_row_end, height, E };
        }
    }

    template <TransitionType Type>
    int terrain_length_from(Column col) const
    {
        int length = 0;
        for (int i = rep(col), last = static_cast<int>(world.size()); i != last; ++i)
        {
            if (type_of(world[i]) == Type)
            {
                ++length;
            }
            else
            {
                break;
            }
        }
        return length;
    }

    template <typename T>
    std::vector<float> transitional_noise(Column col, Height height, Width width, T&& distribution) const
    {
        if (rep(col) == 0)
        {
            return noise_seed(rep(width), std::forward<T>(distribution));
        }
        float previous_noise = static_cast<float>(top_of(retract(col))) / rep(height);
        auto seed = noise_seed(rep(width), std::forward<T>(distribution));
        seed.insert(begin(seed), previous_noise);
        return seed;
    }

    Row top_of(Column col) const
    {
        const auto& data = world[rep(col)];
        switch (type_of(data))
        {
        case TransitionType::Mountains:
            return top_of_internal(as_collection<TransitionType::Mountains>(col));
        case TransitionType::Valley:
            return top_of_internal(as_collection<TransitionType::Valley>(col));
        case TransitionType::Lowlands:
            return top_of_internal(as_collection<TransitionType::Lowlands>(col));
        case TransitionType::Beach:
            return top_of_internal(as_collection<TransitionType::Beach>(col));
        }
        return unit<Row>;
    }

    template <typename T>
    Row top_of_internal(const T& collection) const
    {
        constexpr auto first_non_blank = extend(ElementsOf<T::type>::Blank);
        return collection.collection[rep(first_non_blank)].start;
    }

    std::vector<TransitionVariant> world;
    int current_state_width = 0;
};

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

    void generate_new(Width width, Height height, PerlinNoiseGenerator* noise_generator)
    {
        world_dimensions(width, height);
        WorldHeightGenerator generator;
        generator.generate(width, height, noise_generator);

        computed_pixels.resize(rep(width) * rep(height));
        for (int i = 0; i != rep(height); ++i)
        {
            for (int j = 0; j != rep(width); ++j)
            {
                computed_pixels[j + i * rep(width)] = generator.pixel_for_compute(Row(i), Column(j));
            }
        }
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

            h = static_cast<int>(crust_to_rock);
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
        if (len == .0f)
        {
            return { };
        }
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

struct AABBBox
{
    olc::vi2d center = { };
    Radius radius = { };
};

bool overlap_AABB(const AABBBox& first, const AABBBox& second)
{
    return first.center.x + rep(first.radius) + rep(second.radius) > second.center.x
        && first.center.x < second.center.x + rep(first.radius) + rep(second.radius)
        && first.center.y + rep(first.radius) + rep(second.radius) > second.center.y
        && first.center.y < second.center.y + rep(first.radius) + rep(second.radius);
}

struct PhysicalProperties
{
    float stickyness = 150.f;
    float friction = .85f;
};

class PhysicsPixel
{
public:
    PhysicsPixel(const olc::vi2d& pos, const olc::vf2d& velocity, olc::Pixel color, Radius radius, Weight weight = Weight(1), PhysicalProperties properties = { }):
        pos{ pos }, vel{ velocity }, pixel_color{ color }, r{ radius }, w{ weight }, properties{ properties } { }

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

    void dead(bool b)
    {
        exploded = b;
    }

    bool dead() const
    {
        return exploded;
    }

    const olc::vi2d& position() const
    {
        return pos;
    }

    olc::vi2d& position()
    {
        return pos;
    }

    const olc::vf2d& velocity() const
    {
        return vel;
    }

    olc::vf2d& velocity()
    {
        return vel;
    }

    olc::Pixel color() const
    {
        return pixel_color;
    }

    Radius radius() const
    {
        return r;
    }

    Weight weight() const
    {
        return w;
    }

    bool single_point() const
    {
        return rep(radius()) == 0;
    }

    AABBBox bounding_box() const
    {
        return { position(), radius() };
    }

    static void collide(PhysicsPixel* a, PhysicsPixel* b)
    {
        auto collision_vector = static_cast<olc::vf2d>(b->position() - a->position());
        float distance = collision_vector.mag();
        if (distance == .0f)
        {
            distance = .1f;
        }
        olc::vf2d collision_normal = collision_vector / distance;
        auto relative_velocity = a->velocity() - b->velocity();
        float speed = relative_velocity.dot(collision_normal);
        // These objects are moving away from each other already.
        if (speed < 0)
        {
            return;
        }
        float impulse = 2 * speed / (rep(a->weight()) + rep(b->weight()));
        a->velocity() -= collision_normal * impulse * static_cast<float>(rep(b->weight()));
        b->velocity() += collision_normal * impulse * static_cast<float>(rep(a->weight()));
    }

    static bool collides_with(const PhysicsPixel& a, const PhysicsPixel& b)
    {
        auto angle = a.position() - b.position();
        int distance_2 = angle.mag2();
        return distance_2 < ((rep(a.radius()) + rep(b.radius())) * (rep(a.radius()) + rep(b.radius())));
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
        if (single_point())
        {
            (*world)(point) = color();
        }
        else
        {
            // Taken from DrawCircle.
            int x0 = 0;
            int y0 = rep(radius());
            int d = 3 - 2 * y0;
            auto [x, y] = position();

            auto adhere = [&](int sx, int ex, int ny)
            {
                for (int i = sx; i <= ex; i++)
                {
                    if (world->bounded(Row(ny), Column(i)))
                    {
                        (*world)(Row(ny), Column(i)) = color();
                    }
                }
            };

            while (y0 >= x0)
            {
                // Modified to draw scan-lines instead of edges
                adhere(x - x0, x + x0, y - y0);
                adhere(x - y0, x + y0, y - x0);
                adhere(x - x0, x + x0, y + y0);
                adhere(x - y0, x + y0, y + x0);
                if (d < 0)
                {
                    d += 4 * x0++ + 6;
                }
                else
                {
                    d += 4 * (x0++ - y0--) + 10;
                }
            }
        }
        exploded = true;
    }

    bool can_stick() const
    {
        return velocity().mag2() < (properties.stickyness * properties.stickyness);
    }

    olc::Pixel pixel_color;
    olc::vi2d pos;
    olc::vf2d vel;
    Radius r;
    Weight w;
    olc::vi2d prev_pos = pos;
    PhysicalProperties properties;
    bool exploded = false;
};

enum class Level : int { };
class BoundingBox
{
public:
    BoundingBox(const olc::vi2d& upper_left, Width width, Height height):
        upper_left{ upper_left }, w{ width }, h{ height } { }

    int left() const
    {
        return upper_left.x;
    }

    int top() const
    {
        return upper_left.y;
    }

    int bottom() const
    {
        return top() + rep(height());
    }

    int right() const
    {
        return left() + rep(width());
    }

    Width width() const
    {
        return w;
    }

    Height height() const
    {
        return h;
    }
private:
    olc::vi2d upper_left;
    Width w;
    Height h;
};

class QuadTree
{
    static constexpr int max_depth = 5;
    static constexpr int split_factor = 20;
public:
    QuadTree(const olc::vi2d& upper_left, Width width, Height height, Level level):
        rect{ upper_left, width, height }, level{ level } { }

    void clear()
    {
        objects.clear();
        trees = { }; // clear the existing trees.
    }

    void insert(PhysicsPixel* pixel)
    {
        internal_insert(box_for(*pixel), pixel);
    }

    template <std::invocable<PhysicsPixel*> F>
    void for_each_in(const BoundingBox& box, F&& invocable) const
    {
        int i = index(box);
        if (i != -1 && trees[i])
        {
            trees[i]->for_each_in(box, std::forward<F>(invocable));
        }
        for (PhysicsPixel* pixel : objects)
        {
            std::forward<F>(invocable)(pixel);
        }
    }

    static BoundingBox box_for(const PhysicsPixel& pixel)
    {
        int r = rep(pixel.radius());
        return { pixel.position(), Width(r), Height(r) };
    }

    std::vector<BoundingBox> all_boxes() const
    {
        std::vector<BoundingBox> boxes;
        internal_all_boxes(&boxes);
        return boxes;
    }
private:
    void internal_all_boxes(std::vector<BoundingBox>* boxes) const
    {
        boxes->push_back(rect);

        for (const auto& tree : trees)
        {
            if (tree)
            {
                tree->internal_all_boxes(boxes);
            }
        }
    }

    void internal_insert(const BoundingBox& box, PhysicsPixel* pixel)
    {
        if (trees[0])
        {
            int i = index(box);
            if (i != -1)
            {
                trees[i]->internal_insert(box, pixel);
                return;
            }
        }

        objects.push_back(pixel);

        if (objects.size() > split_factor && rep(level) < max_depth)
        {
            if (!trees[0])
            {
                split();
            }

            int i = 0;
            while (i < static_cast<int>(objects.size()))
            {
                BoundingBox object_box = box_for(*objects[i]);
                int idx = index(object_box);
                if (idx != -1)
                {
                    trees[idx]->internal_insert(object_box, objects[i]);
                    objects.erase(begin(objects) + i);
                }
                else
                {
                    ++i;
                }
            }
        }
    }

    void split()
    {
        int sub_width = rep(rect.width()) / 2;
        int sub_height = rep(rect.height()) / 2;
        int x = rect.left();
        int y = rect.top();
        int next_level = rep(level) + 1;

        trees[0] = std::make_unique<QuadTree>(olc::vi2d{ x + sub_width, y },              Width(sub_width), Height(sub_height), Level(next_level));
        trees[1] = std::make_unique<QuadTree>(olc::vi2d{ x,             y },              Width(sub_width), Height(sub_height), Level(next_level));
        trees[2] = std::make_unique<QuadTree>(olc::vi2d{ x,             y + sub_height }, Width(sub_width), Height(sub_height), Level(next_level));
        trees[3] = std::make_unique<QuadTree>(olc::vi2d{ x + sub_width, y + sub_height }, Width(sub_width), Height(sub_height), Level(next_level));
    }

    int index(const BoundingBox& box) const
    {
        int index = -1;
        float vert_mid = rect.left() + static_cast<float>(rep(rect.width())) / 2.f;
        float horiz_mid = rect.top() + static_cast<float>(rep(rect.height())) / 2.f;

        auto is_top = [&]
        {
            return box.top() < horiz_mid && box.bottom() < horiz_mid;
        };
        auto is_bottom = [&]
        {
            return box.top() > horiz_mid;
        };

        if (box.left() <= vert_mid && box.right() <= vert_mid)
        {
            if (is_top())
            {
                index = 1;
            }
            else if (is_bottom())
            {
                index = 2;
            }
        }
        else if (box.left() >= vert_mid)
        {
            if (is_top())
            {
                index = 0;
            }
            else if (is_bottom())
            {
                index = 3;
            }
        }

        return index;
    }

    std::array<std::unique_ptr<QuadTree>, max_depth> trees;
    std::vector<PhysicsPixel*> objects;
    BoundingBox rect;
    const Level level;
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

        if (all_collisions)
        {
            build_quadtree(world);
        }

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

        // If we are doing extra interactions, do them.
        if (all_collisions)
        {
#if 1
            // Note: parallel for_each?
            std::for_each(std::execution::par, begin(pixels_objects), end(pixels_objects),
                [&](PhysicsPixel& pixel)
                {
                    intersect_objects(world, &pixel);
                });
#else
            for (auto& pixel : pixels_objects)
            {
                intersect_objects(world, &pixel);
            }
#endif
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

    void more_collisions(bool b)
    {
        all_collisions = b;
    }

    bool more_collisions() const
    {
        return all_collisions;
    }

    QuadTree* current_quad_tree() const
    {
        return quad_tree.get();
    }
private:
    void build_quadtree(World* world)
    {
        quad_tree = nullptr;
        quad_tree = std::make_unique<QuadTree>(olc::vi2d{ 0, 0 }, world->width(), world->height(), Level(0));
        for (PhysicsPixel& pixel : pixels_objects)
        {
            if (!pixel.dead())
            {
                quad_tree->insert(&pixel);
            }
        }
    }

    void intersect_objects(World*, PhysicsPixel* pixel)
    {
        assert(quad_tree != nullptr);
        quad_tree->for_each_in(QuadTree::box_for(*pixel),
                            [&](PhysicsPixel* other)
                            {
                                if (other == pixel)
                                {
                                    return;
                                }

                                if (other->dead())
                                {
                                    return;
                                }

                                if (overlap_AABB(pixel->bounding_box(), other->bounding_box())
                                    && PhysicsPixel::collides_with(*pixel, *other))
                                {
                                    PhysicsPixel::collide(pixel, other);
                                }
                            });
    }

    int left_over_time = 0;
    std::vector<PhysicsPixel> pixels_objects;
    std::unique_ptr<QuadTree> quad_tree;
    bool all_collisions = true;
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
                                                  (*world)(Row(y), Column(x)),
                                                  Radius(0),
                                                  Weight(1),
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

        if (GetKey(olc::Key::P).bReleased)
        {
            physics_engine.more_collisions(!physics_engine.more_collisions());
        }

        if (GetKey(olc::Key::G).bReleased)
        {
            old_generated = !old_generated;
            if (old_generated)
            {
                noise_generator.seed_generator(noise_seed(rep(screen_info.width)));
            }
        }

        if (changed)
        {
            Clear(olc::BLACK);
            if (old_generated)
            {
                world.generate_world(screen_info.width, screen_info.height, noise_generator);
            }
            else
            {
                world.generate_new(screen_info.width, screen_info.height, &noise_generator);
            }
        }

        physics_engine.update(elapsed_time, &world);

        if (GetMouse(1).bHeld)
        {
            std::uniform_real_distribution<float> dis_velocity_x{ 50.f, 1500.f };
            std::uniform_real_distribution<float> dis_velocity_y{ 50.f, 500.f };
            std::uniform_real_distribution<float> dis_stickyness{ 150.f, 1500.f };
            std::uniform_real_distribution<float> dis_friction{ .10f, .85f };
            std::uniform_int_distribution<int> dis_color{ 0, 255 };
            std::uniform_int_distribution<int> dis_radius{ 0, 10 };
            for (int i = 0; i != 10; ++i)
            {
                olc::vf2d velocity{ random_generator().generate(dis_velocity_x), random_generator().generate(dis_velocity_y) };
                olc::Pixel color {
                    static_cast<uint8_t>(random_generator().generate(dis_color)),
                    static_cast<uint8_t>(random_generator().generate(dis_color)),
                    static_cast<uint8_t>(random_generator().generate(dis_color)) };
                auto radius = random_generator().generate(dis_radius);
                auto weight = radius * 2 + 1;
                physics_engine.add(PhysicsPixel{ { GetMouseX(), GetMouseY() },
                                                    velocity,
                                                    color,
                                                    Radius(radius),
                                                    //Radius(0),
                                                    Weight(weight),
                                                    { .stickyness = random_generator().generate(dis_stickyness),
                                                      .friction = random_generator().generate(dis_friction) } });
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

                if (pixel.single_point())
                {
                    Draw(pixel.position(), pixel.color());
                }
                else
                {
                    FillCircle(pixel.position(), rep(pixel.radius()), pixel.color());
                }
            }
        }

        if (GetKey(olc::Key::Q).bHeld)
        {
            if (QuadTree* tree = physics_engine.current_quad_tree())
            {
                auto boxes = tree->all_boxes();
                for (const BoundingBox& box : boxes)
                {
                    constexpr olc::Pixel color = olc::RED;
                    // Top line
                    DrawLine({ box.left(), box.top() }, { box.right(), box.top() }, color);
                    // Right line
                    DrawLine({ box.right(), box.top() }, { box.right(), box.bottom() }, color);
                    // Bottom line
                    DrawLine({ box.right(), box.bottom() }, { box.left(), box.bottom() }, color);
                    // Left line
                    DrawLine({ box.left(), box.bottom() }, { box.left(), box.top() }, color);
                }
            }
        }

        {
            std::stringstream ss;
            ss << "All collisions (" << (physics_engine.more_collisions() ? "on" : "off") << ") Particle Count: " << alive_particles;
            DrawString({ 10, 20 }, ss.str());
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

    ScreenInfo screen_info;
    World world;
    PerlinNoiseGenerator noise_generator;
    PhysicsEngine physics_engine;

    // Debug info stuff
    std::vector<std::pair<olc::vi2d, olc::vi2d>> computed_normals;
    std::string display_text;
    bool old_generated = true; // Use old random generated terrain.
};

int main()
{
    Game game;
    if (game.Construct({ Width(1540), Height(978), PixelWidth(1), PixelHeight(1) }))
    {
        game.Start();
    }
}