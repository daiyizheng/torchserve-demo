package com.dyz.onnx;
import java.io.File;

import java.nio.LongBuffer;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import com.robrua.nlp.bert.FullTokenizer;

class Tokens {
    public long[] ids;
    public long[] mask;
    public long[] types;
}

class Tokenizer {
    private FullTokenizer tokenizer;

    public Tokenizer(String path) {
        File vocab = new File(path);
        this.tokenizer = new FullTokenizer(vocab, true);
    }

public Tokens tokenize(String text) {
    // Build list of tokens
    List<String> tokensList = new ArrayList();
    tokensList.add("[CLS]");
    tokensList.addAll(Arrays.asList(tokenizer.tokenize(text)));
    tokensList.add("[SEP]");

    int[] ids = tokenizer.convert(tokensList.toArray(new String[0]));

        Tokens tokens = new Tokens();

        // input ids
        tokens.ids = Arrays.stream(ids).mapToLong(i -> i).toArray();

        // attention mask
        tokens.mask = new long[ids.length];
        Arrays.fill(tokens.mask, 1);

        // token type ids
        tokens.types = new long[ids.length];
        Arrays.fill(tokens.types, 0);

        return tokens;
    }
}

class Inference {
    private Tokenizer tokenizer;
    private OrtEnvironment env;
    private OrtSession session;

    public Inference(String model) throws Exception {
        this.tokenizer = new Tokenizer("D:\\project\\after-web\\onnxJava\\src\\main\\resources\\vocab.txt");
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(model, new OrtSession.SessionOptions());
    }

    public float[][] predict(String text) throws Exception {
        Tokens tokens = this.tokenizer.tokenize(text);

        Map<String, OnnxTensor> inputs = new HashMap<String, OnnxTensor>();
        inputs.put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.ids),  new long[]{1, tokens.ids.length}));
        inputs.put("attention_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.mask),  new long[]{1, tokens.mask.length}));
        inputs.put("token_type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.types),  new long[]{1, tokens.types.length}));

        return (float[][])session.run(inputs).get(0).getValue();
    }
}

class Vectors {
    public static double similarity(float[] v1, float[] v2) {
        double dot = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int x = 0; x < v1.length; x++) {
            dot += v1[x] * v2[x];
            norm1 += Math.pow(v1[x], 2);
            norm2 += Math.pow(v2[x], 2);
        }

        return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    public static float[] softmax(float[] input) {
        double[] t = new double[input.length];
        double sum = 0.0;

        for (int x = 0; x < input.length; x++) {
            double val = Math.exp(input[x]);
            sum += val;
            t[x] = val;
        }

        float[] output = new float[input.length];
        for (int x = 0; x < output.length; x++) {
            output[x] = (float) (t[x] / sum);
        }

        return output;
    }
}


public class OnnxDemo {
    public void textClassify() throws Exception {
        Inference inference = new Inference("D:\\project\\after-web\\onnxJava\\src\\main\\resources\\text-classify.onnx");
        float[][] v1 = inference.predict("I am glad");
        System.out.println(Arrays.toString(Vectors.softmax(v1[0])));
    }

    public void embeddings() throws Exception {
        Inference inference = new Inference("D:\\project\\after-web\\onnxJava\\src\\main\\resources\\embeddings.onnx");
        float[][] v1 = inference.predict("I am happy");
        float[][] v2 = inference.predict("I am glad");
        System.out.println(Vectors.similarity(v1[0], v2[0]));
    }

    public static void main(String[] args) throws Exception {
        OnnxDemo onnxDemo = new OnnxDemo();
        onnxDemo.embeddings();
        onnxDemo.textClassify();
    }
}


