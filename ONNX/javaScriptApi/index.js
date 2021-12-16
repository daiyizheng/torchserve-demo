const ort = require('onnxruntime-node');
const { promisify } = require('util');
const { Tokenizer } = require("tokenizers/dist/bindings/tokenizer");

function sigmoid(data) {
    return data.map(x => 1 / (1 + Math.exp(-x)))
}

function softmax(data) {
    return data.map(x => Math.exp(x) / (data.map(y => Math.exp(y))).reduce((a,b) => a+b))
}

function similarity(v1, v2) {
    let dot = 0.0;
    let norm1 = 0.0;
    let norm2 = 0.0;

    for (let x = 0; x < v1.length; x++) {
        dot += v1[x] * v2[x];
        norm1 += Math.pow(v1[x], 2);
        norm2 += Math.pow(v2[x], 2);
    }

    return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

function tokenizer(path) {
    let tokenizer = Tokenizer.fromFile(path);
    return promisify(tokenizer.encode.bind(tokenizer));
}

async function predict(session, text) {
    try {
        // Tokenize input
        let encode = tokenizer("bert/tokenizer.json");
        let output = await encode(text);

        let ids = output.getIds().map(x => BigInt(x))
        let mask = output.getAttentionMask().map(x => BigInt(x))
        let tids = output.getTypeIds().map(x => BigInt(x))

        // Convert inputs to tensors
        let tensorIds = new ort.Tensor('int64', BigInt64Array.from(ids), [1, ids.length]);
        let tensorMask = new ort.Tensor('int64', BigInt64Array.from(mask), [1, mask.length]);
        let tensorTids = new ort.Tensor('int64', BigInt64Array.from(tids), [1, tids.length]);

        let inputs = null;
        if (session.inputNames.length > 2) {
            inputs = { input_ids: tensorIds, attention_mask: tensorMask, token_type_ids: tensorTids};
        }
        else {
            inputs = { input_ids: tensorIds, attention_mask: tensorMask};
        }

        return await session.run(inputs);
    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

async function main() {
    let args = process.argv.slice(2);
    if (args.length > 1) {
        // Run sentence embeddings
        const session = await ort.InferenceSession.create('./embeddings.onnx');

        let v1 = await predict(session, args[0]);
        let v2 = await predict(session, args[1]);

        // Unpack results
        v1 = v1.embeddings.data;
        v2 = v2.embeddings.data;

        // Print similarity
        console.log(similarity(Array.from(v1), Array.from(v2)));
    }
    else {
        // Run text classifier
        const session = await ort.InferenceSession.create('./text-classify.onnx');
        let results = await predict(session, args[0]);

        // Normalize results using softmax and print
        console.log(softmax(results.logits.data));
    }
}

main();