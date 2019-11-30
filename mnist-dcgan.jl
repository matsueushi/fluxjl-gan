using BSON
using Base.Iterators: partition
using DelimitedFiles
using Flux
using Flux.Data.MNIST
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, glorot_normal, gradient, grad
using Images
using Statistics
using Printf


mutable struct DCGAN
    noise_dim::Int64
    channels::Int64
    batch_size::Int64
    epochs::Int64

    generator::Chain
    discriminator::Chain

    generator_optimizer
    discriminator_optimizer

    data::Vector{<: AbstractArray{Float32, 4}}

    animation_size::Pair{Int64, Int64}
    animation_noise::AbstractMatrix{Float32}

    train_steps::Int64
    verbose_freq::Int64
    generator_loss_hist::Vector{Float32}
    discriminator_loss_hist::Vector{Float32}
end

function DCGAN(; image_vector::Vector{<: AbstractMatrix}, noise_dim::Int64, channels::Int64, batch_size::Int64, epochs::Int64,
    generator::Chain, discriminator::Chain,
    animation_size::Pair{Int64, Int64}, verbose_freq::Int64)

    @assert (channels == 1) || (channels == 3)

    data = [reshape(reduce(hcat, channelview.(xs)), 28, 28, 1, :) for xs in partition(image_vector, batch_size)]
    data = [2f0 .* gpu(Float32.(xs)) .- 1f0 for xs in data]

    animation_noise = randn(Float32, noise_dim, prod(animation_size)) |> gpu

    DCGAN(noise_dim, channels, batch_size, epochs, generator, discriminator, ADAM(0.0001f0), ADAM(0.0001f0), data, 
        animation_size, animation_noise, 0, verbose_freq, Vector{Float32}(), Vector{Float32}())
end

function generator_loss(fake_output)
    loss = mean(logitbinarycrossentropy.(fake_output, 1f0))
end

function discriminator_loss(real_output, fake_output)
    real_loss = mean(logitbinarycrossentropy.(real_output, 1f0))
    fake_loss = mean(logitbinarycrossentropy.(fake_output, 0f0))
    loss = 0.5f0 * (real_loss +  fake_loss)
    return loss
end

function convert_to_image(image_array::Matrix{Float32}, channels::Int64)
    image_array = @. (image_array + 1f0) / 2f0
    if channels == 1
        return Gray.(image_array)
    else
        return colorview(RGB, image_array)
    end
end

function save_fake_image(dcgan::DCGAN)
    @eval Flux.istraining() = false
    fake_images = dcgan.generator(dcgan.animation_noise)
    @eval Flux.istraining() = true
    h, w, _, _ = size(fake_images)
    rows, cols = dcgan.animation_size.first, dcgan.animation_size.second
    tile_image = Matrix{Float32}(undef, h * rows, w * cols)
    for n in 0:prod(dcgan.animation_size) - 1
        j = n ÷ rows
        i = n % cols
        tile_image[j * h + 1:(j + 1) * h, i * w + 1:(i + 1) * w] = fake_images[:, :, :, n + 1] |> cpu
    end
    image = convert_to_image(tile_image, dcgan.channels)
    save(@sprintf("animation/steps_%06d.png", dcgan.train_steps), image)
end

function train_discriminator!(dcgan::DCGAN, batch::AbstractArray{Float32, 4})
    noise = randn(Float32, dcgan.noise_dim, dcgan.batch_size) |> gpu
    fake_input = dcgan.generator(noise)
    fake_output = dcgan.discriminator(fake_input)

    real_output = dcgan.discriminator(batch)

    disc_loss = discriminator_loss(real_output, fake_output)
    disc_grad = gradient(()->disc_loss, Flux.params(dcgan.discriminator))
    update!(dcgan.discriminator_optimizer, Flux.params(dcgan.discriminator), disc_grad)
    
    # zero out generator gradient
    # https://github.com/FluxML/model-zoo/pull/111
    grad.(Flux.params(dcgan.generator)) .= 0f0
    return disc_loss
end

function train_generator!(dcgan::DCGAN, batch::AbstractArray{Float32, 4})
    noise = randn(Float32, dcgan.noise_dim, dcgan.batch_size) |> gpu
    fake_input = dcgan.generator(noise)
    fake_output = dcgan.discriminator(fake_input)

    gen_loss = generator_loss(fake_output)
    gen_grad = gradient(()->gen_loss, Flux.params(dcgan.generator))
    update!(dcgan.generator_optimizer, Flux.params(dcgan.generator), gen_grad)
    return gen_loss
end

function train!(dcgan::DCGAN)
    for ep in 1:dcgan.epochs
        @info "epoch $ep"
        for batch in dcgan.data
            disc_loss = train_discriminator!(dcgan, batch)
            gen_loss = train_generator!(dcgan, batch)

            if dcgan.train_steps % dcgan.verbose_freq == 0
                push!(dcgan.discriminator_loss_hist, disc_loss)
                push!(dcgan.generator_loss_hist, gen_loss)
                @info("Train step $(dcgan.train_steps), Discriminator loss: $(disc_loss), Generator loss: $(gen_loss)")
                # create fake images for animation
                save_fake_image(dcgan)
            end
            dcgan.train_steps += 1
        end
    end
end


function main()
    @info "start..."
    if !isdir("animation")
        mkdir("animation")
    end

    if !isdir("result")
        mkdir("result")
    end

    noise_dim = 100
    channels = 1

    generator = Chain(
        Dense(noise_dim, 7 * 7 * 256; initW = glorot_normal),
        BatchNorm(7 * 7 * 256, relu),
        x->reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128; init = glorot_normal, stride = 1, pad = 2),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; init = glorot_normal, stride = 2, pad = 1),
        BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => channels, tanh; init = glorot_normal, stride = 2, pad = 1),
        ) |> gpu

    discriminator =  Chain(
        Conv((4, 4), channels => 64, leakyrelu; init = glorot_normal, stride = 2, pad = 1),
        Dropout(0.25),
        Conv((4, 4), 64 => 128, leakyrelu; init = glorot_normal, stride = 2, pad = 1),
        Dropout(0.25), 
        x->reshape(x, 7 * 7 * 128, :),
        # drop sigmoid, and use logitbinarycrossentropy (it is more numerically stable)
        # https://github.com/FluxML/Flux.jl/issues/914
        Dense(7 * 7 * 128, 1; initW = glorot_normal)) |> gpu 

    dcgan = DCGAN(; image_vector = MNIST.images(), noise_dim = noise_dim, 
        channels = channels, batch_size = 128, epochs = 30,
        generator = generator, discriminator = discriminator,
        animation_size = 6=>6, verbose_freq = 100)
    train!(dcgan)

    open("result/discriminator_loss.txt", "w") do io
        writedlm(io, dcgan.discriminator_loss_hist)
    end

    open("result/generator_loss.txt", "w") do io
        writedlm(io, dcgan.generator_loss_hist)
    end

    wts_generator = Flux.params(dcgan.generator)
    wts_discriminator = Flux.params(dcgan.discriminator)
    BSON.@save "result/mnist-dcgan-generator.bson" wts_generator
    BSON.@save "result/mnist-dcgan-discriminator.bson" wts_discriminator
end

main()