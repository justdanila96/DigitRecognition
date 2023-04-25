namespace NeuralNetwork

open ILGPU
open ILGPU.Algorithms
open ILGPU.Runtime
open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open SixLabors.ImageSharp
open SixLabors.ImageSharp.PixelFormats

type DatasetItem(filePath: string) =

    let file = new FileInfo(filePath)

    member val Number: int =
        let regex = new Regex("^\\d{6}\\-num(\\d{1})\\.png$")
        let matches = regex.Matches(file.Name)
        let number = matches[0].Groups[1].Value
        Int32.Parse(number)

    member this.GetPixelsData() =
        use img = Image.Load<Byte4>(file.FullName)

        [| for y = 0 to img.Height - 1 do
               for x = 0 to img.Width - 1 do
                   let pixel = img[x, y]
                   let pixelVal = pixel.PackedValue &&& 0xFFu |> Convert.ToSingle
                   yield pixelVal / 255f |]

type Layer(size: int, nextSize: int, accelerator: Accelerator) =
    static member private RandomVal: float32 = 2f * Random.Shared.NextSingle() - 1f
    member val Size: int = size with get
    member val Neurons = size |> accelerator.Allocate1D<float32> with get
    member val Biases = [| for _ in 1..size -> Layer.RandomVal |] |> accelerator.Allocate1D with get

    member val Weights =
        Array2D.init size nextSize (fun _ _ -> Layer.RandomVal)
        |> accelerator.Allocate2DDenseX with get

type IBrainSettings =
    abstract member PathToDataset: string with get
    abstract member EpochsQuantity: int with get
    abstract member BatchSize: int with get

type Brain(accelerator: Accelerator, rate: float32, sizes: int array) =

    member val LearningRate = rate with get

    member val Layers =
        let layers = new LinkedList<Layer>()

        for i = 0 to sizes.Length - 1 do
            let nextSize = if i < sizes.Length - 1 then sizes[i + 1] else 1
            new Layer(sizes[i], nextSize, accelerator) |> layers.AddLast |> ignore

        layers with get

    member this.FeedForward(inputs: float32 array, accelerator: Accelerator) =
        this.Layers.First.Value.Neurons.CopyFromCPU(inputs)

        let feedForward =
            accelerator.LoadAutoGroupedStreamKernel(KernelCommands.FeedForward)

        let mutable layer = this.Layers.First.Next

        while layer <> null do
            feedForward.Invoke(
                layer.Value.Size,
                layer.Previous.Value.Weights.View,
                layer.Previous.Value.Neurons.View,
                layer.Value.Biases.View,
                layer.Value.Neurons.View
            )

            layer <- layer.Next

        this.Layers.Last.Value.Neurons.GetAsArray1D()

    member this.BackPropagation(inTargets: float32 array, accelerator: Accelerator) =

        let funErrors = accelerator.LoadAutoGroupedStreamKernel(KernelCommands.CalcErrors)

        let funGradients =
            accelerator.LoadAutoGroupedStreamKernel(KernelCommands.CalcGradients)

        let funDeltas = accelerator.LoadAutoGroupedStreamKernel(KernelCommands.CalcDeltas)

        let funNextErrors =
            accelerator.LoadAutoGroupedStreamKernel(KernelCommands.CalcNextErrors)

        let funNewWeights =
            accelerator.LoadAutoGroupedStreamKernel(KernelCommands.CalcNewWeights)

        let funBiases = accelerator.LoadAutoGroupedStreamKernel(KernelCommands.CalcBiases)

        use targets = accelerator.Allocate1D(inTargets)
        let mutable errors = accelerator.Allocate1D<float32>(targets.Length)
        funErrors.Invoke(inTargets.Length, targets.View, this.Layers.Last.Value.Neurons.View, errors.View)

        let mutable layer = this.Layers.Last.Previous

        while layer <> null do
            use gradients = accelerator.Allocate1D<float32>(layer.Next.Value.Size)

            funGradients.Invoke(
                layer.Next.Value.Size,
                errors.View,
                layer.Next.Value.Neurons.View,
                this.LearningRate,
                gradients.View
            )

            let deltaSize = new LongIndex2D(layer.Next.Value.Size, layer.Value.Size)
            use deltas = accelerator.Allocate2DDenseX<float32>(&deltaSize)
            funDeltas.Invoke(deltaSize.ToIntIndex(), gradients.View, layer.Value.Neurons.View, deltas.View)

            use nextErrors = accelerator.Allocate1D<float32>(layer.Value.Size)
            funNextErrors.Invoke(layer.Value.Size, layer.Value.Weights.View, errors.View, nextErrors.View)

            errors.Dispose()
            errors <- accelerator.Allocate1D(nextErrors.GetAsArray1D())

            let weightSize = new Index2D(layer.Value.Size, layer.Next.Value.Size)
            funNewWeights.Invoke(weightSize, deltas.View, layer.Value.Weights.View)
            funBiases.Invoke(layer.Next.Value.Size, gradients.View, layer.Next.Value.Biases.View)

            layer <- layer.Previous

        errors.Dispose()

    member this.Train(settings: IBrainSettings, accelerator: Accelerator) =

        let dataset =
            settings.PathToDataset
            |> Directory.GetFiles
            |> Array.map (fun file -> new DatasetItem(file))

        for _ = 1 to settings.EpochsQuantity do
            for _ = 1 to settings.BatchSize do

                let itemIndex = Random.Shared.Next(dataset.Length)
                let targets = Array.zeroCreate 10
                let digit = dataset[itemIndex].Number
                targets[digit] <- 1f
                let inputs = dataset[itemIndex].GetPixelsData()
                let _ = this.FeedForward(inputs, accelerator)
                this.BackPropagation(targets, accelerator)

    member this.Recognize(binData: byte array, pixelSize: int, accelerator: Accelerator) =

        //let inputData =
        //    binData
        //    |> Array.chunkBySize pixelSize
        //    |> Array.map (fun chunk ->
        //        let chunkf = chunk |> Array.map (fun item -> item |> Convert.ToSingle)
        //        (chunkf[0] + chunkf[1] + chunkf[2]) / 3f)

        let inputData =
            binData
            |> Array.chunkBySize pixelSize
            |> Array.map (fun chunk -> chunk |> Array.take 3 |> Array.averageBy (fun i -> Convert.ToSingle(i)))

        let outputs = this.FeedForward(inputData, accelerator)
        let maxVal = outputs |> Array.max
        outputs |> Array.findIndex (fun item -> item = maxVal)

    member this.Save(filePath: string) =

        let GetBytesVector (buffer: MemoryBuffer1D<float32, Stride1D.Dense>) =

            let bufferData = buffer.GetAsArray1D()
            let bytes: byte array = Array.zeroCreate (bufferData.Length * sizeof<float32>)
            Buffer.BlockCopy(bufferData, 0, bytes, 0, bytes.Length)
            bytes

        let GetBytesMatrix (buffer: MemoryBuffer2D<float32, Stride2D.DenseX>) =

            let bufferData = buffer.GetAsArray2D()
            let bytes: byte array = Array.zeroCreate (bufferData.Length * sizeof<float32>)
            Buffer.BlockCopy(bufferData, 0, bytes, 0, bytes.Length)
            bytes

        use fileStream = new FileStream(filePath, FileMode.Create)
        use binWriter = new BinaryWriter(fileStream)
        let mutable layer = this.Layers.First

        while layer <> null do

            binWriter.Write(layer.Value.Neurons |> GetBytesVector)
            binWriter.Write(layer.Value.Biases |> GetBytesVector)
            binWriter.Write(layer.Value.Weights |> GetBytesMatrix)
            layer <- layer.Next

    member this.Read(filePath: string) =

        let ReadVectorData (buffer: MemoryBuffer1D<float32, Stride1D.Dense>) (reader: BinaryReader) =

            let binData = reader.ReadBytes(buffer.View.IntExtent.Size * sizeof<float32>)
            let data: float32 array = Array.zeroCreate (buffer.View.IntLength)
            Buffer.BlockCopy(binData, 0, data, 0, binData.Length)
            buffer.CopyFromCPU(data)

        let ReadMatrixData (buffer: MemoryBuffer2D<float32, Stride2D.DenseX>) (reader: BinaryReader) =

            let binData = reader.ReadBytes(buffer.View.IntExtent.Size * sizeof<float32>)

            let data: float32 array2d =
                Array2D.zeroCreate buffer.View.IntExtent.X buffer.View.IntExtent.Y

            Buffer.BlockCopy(binData, 0, data, 0, binData.Length)
            buffer.CopyFromCPU(data)

        use fileStream = new FileStream(filePath, FileMode.Open)
        use binReader = new BinaryReader(fileStream)
        let mutable layer = this.Layers.First

        while layer <> null do

            ReadVectorData layer.Value.Neurons binReader
            ReadVectorData layer.Value.Biases binReader
            ReadMatrixData layer.Value.Weights binReader
            layer <- layer.Next
