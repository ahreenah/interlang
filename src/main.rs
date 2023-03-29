use std::alloc::System;
use std::process;
use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use std::mem::ManuallyDrop;
#[macro_use]
extern crate log;

#[derive(Debug, Clone)]
pub enum Token {
    Keyword(String, usize),
    Name(String, usize),
    Number(f64, usize),
    MathSign(String, usize),
    SpecialSign(char, usize),
    Bracket(char, usize),
}

#[derive(Debug, Clone)]
pub struct TokenTree {
    token: Token,
    children: Vec<Token>,
}

impl TokenTree {
    pub fn new(token: Token) -> Self {
        let mut tokenTree = TokenTree {
            token: token,
            children: Vec::new(),
        };
        tokenTree
    }
}

#[derive(Debug, Clone)]
pub struct TokenTreeRec {
    token: Token,
    children: Vec<TokenTreeRec>,
}

impl TokenTreeRec {
    pub fn new(token: Token) -> Self {
        let mut tokenTree = TokenTreeRec {
            token: token,
            children: Vec::new(),
        };
        tokenTree
    }
}

pub struct Lexer<'a> {
    input: &'a str,
    chars: Box<dyn Iterator<Item = char> + 'a>,
    current_char: Option<char>,
    nest_level: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            input,
            chars: Box::new(input.chars()),
            current_char: None,
            nest_level: 1,
        };
        lexer.advance();
        lexer
    }

    fn advance(&mut self) {
        self.current_char = self.chars.next();
    }

    fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();

        let keywords = [
            "var",
            "func",
            "let",
            "return",
            "if",
            "else",
            "for",
            "while",
            "switch",
            "case",
        ];

        while let Some(current_char) = self.current_char {
            match current_char {
                'a'..='z' | 'A'..='Z' => {
                    let name = self.parse_name();

                    if keywords.contains(&name.as_str()) {
                        tokens.push(Token::Keyword(name, self.nest_level));
                    } else {
                        tokens.push(Token::Name(name, self.nest_level));
                    }

                    // tokens.push(Token::Name(name, self.nest_level));
                }
                '0'..='9' => {
                    let number = self.parse_number();
                    tokens.push(Token::Number(number, self.nest_level));
                }
                '+' | '-' | '*' | '/' | '=' | '>' | '<' | '!' => {
                    let mut name = current_char.to_string();
                    self.advance();
                    if ((name=='<'.to_string()) || (name=='>'.to_string()) || (name=='='.to_string()) || (name=='!'.to_string())){
                        if let Some(current_char) = self.current_char {
                            if current_char == '=' {
                                name.push('=');
                            }
                            self.advance();
                        }
                    }

                    tokens.push(Token::MathSign(name, self.nest_level));
                    // let math_sign = self.parse_math_sign();
                    // tokens.push(Token::MathSign(math_sign, self.nest_level));
                    // self.advance();
                },
                '.' => {
                    tokens.push(Token::MathSign(current_char.to_string(), self.nest_level));
                    self.advance();
                }
                ';' => {
                    tokens.push(Token::SpecialSign(current_char, self.nest_level));
                    self.advance();
                }

                '(' | '[' | '{' => {
                    // self.nest_level += 1;
                    tokens.push(Token::Bracket(current_char, self.nest_level));
                    self.nest_level += 1;
                    self.advance();
                }
                ')' | ']' | '}' => {
                    // self.nest_level -= 1;
                    // tokens.push(Token::Bracket(current_char, self.nest_level));
                    self.nest_level -= 1;
                    self.advance();
                }

                _ => {
                    self.advance();
                }
            }
        }

        tokens
    }

    fn parse_name(&mut self) -> String {
        let mut name = String::new();

        while let Some(c) = self.current_char {
            if c.is_alphabetic() || c.is_digit(10) {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }

        name
    }

    fn parse_number(&mut self) -> f64 {
        let mut number = String::new();

        while let Some(c) = self.current_char {
            if c.is_digit(10) || c == '.' {
                number.push(c);
                self.advance();
            } else {
                break;
            }
        }

        number.parse::<f64>().unwrap()
    }

    fn parse_math_sign(&mut self) -> String {
        let mut name = String::new();
        if let Some(c) = self.current_char {
            if((c=='>' ) || (c=='<') || (c=='=')){
                name.push(c);
                self.advance();
                if let Some(c2) = self.current_char {
                    // ended line
                    if((name==">".to_string()) || (name=="<".to_string()) && (c2=='=')){
                        name.push(c2);
                        self.advance()
                    }
                }
            }
        }
        name

        // while let Some(c) = self.current_char {
        //     if c.is_alphabetic() || c.is_digit(10) {
        //         name.push(c);
        //         self.advance();
        //     } else {
        //         break;
        //     }
        // }

        // name
    }
}

fn get_nest_level(token: &Token) -> usize {
    match token {
        Token::Keyword(_, level) => *level,
        Token::Name(_, level) => *level,
        Token::Number(_, level) => *level,
        Token::MathSign(_, level) => *level,
        Token::SpecialSign(_, level) => *level,
        Token::Bracket(_, level) => *level,
    }
}

fn get_name(token: &Token) -> String {
    match token {
        Token::Keyword(name, level) => name.to_string(),
        Token::Name(name, level) => name.to_string(),
        Token::Number(_, level) => "".to_string(),
        Token::MathSign(name, level) => name.to_string(),
        Token::SpecialSign(name, level) => name.to_string(),
        Token::Bracket(name, level) => name.to_string(),
    }
}

fn get_number(token: &Token) -> f64 {
    match token {
        Token::Keyword(name, level) => 0.0,
        Token::Name(name, level) => 0.0,
        Token::Number(v, level) => *v,
        Token::MathSign(name, level) => 0.0,
        Token::SpecialSign(name, level) => 0.0,
        Token::Bracket(name, level) => 0.0,
    }
}

fn get_type(token: &Token) -> String {
    match token {
        Token::Keyword(name, level) => "Keyword".to_string(),
        Token::Name(name, level) => "Name".to_string(),
        Token::Number(_, level) => "Number".to_string(),
        Token::MathSign(name, level) => "MathSign".to_string(),
        Token::SpecialSign(name, level) => "SpeicalSign".to_string(),
        Token::Bracket(name, level) => "Bracket".to_string(),
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    token: Token,
    children: Vec<Node>,
}

impl Node {
    fn new(token: Token) -> Self {
        Node {
            token,
            children: Vec::new(),
        }
    }

    fn add_child(&mut self, child: Node) {
        self.children.push(child);
    }
}

fn build_tree(tokens: &[Token], parent_level: usize) -> Node {
    let mut root = Node::new(Token::Keyword("Code".to_string(), parent_level));
    let mut index = 0;

    while index < tokens.len() {
        let token = &tokens[index];
        let token_level = get_nest_level(token);

        if token_level == parent_level + 1 {
            let new_node = Node::new(token.clone());
            root.add_child(new_node);

            if let Token::Bracket(c, _) = token {
                if *c == '(' || *c == '[' || *c == '{' {
                    let closing_bracket_index = find_closing_bracket(tokens, index);
                    let inner_tokens = &tokens[index + 1..closing_bracket_index];
                    let subtree = build_tree(inner_tokens, token_level);
                    root.children.last_mut().unwrap().children = subtree.children;
                    index = closing_bracket_index;
                }
            }
        }

        index += 1;
    }

    root
}

fn find_closing_bracket(tokens: &[Token], opening_bracket_index: usize) -> usize {
    let opening_bracket = match &tokens[opening_bracket_index] {
        Token::Bracket(c, _) => *c,
        _ => panic!("Token is not a bracket"),
    };
    let closing_bracket = match opening_bracket {
        '(' => ')',
        '[' => ']',
        '{' => '}',
        _ => panic!("Invalid opening bracket"),
    };
    let mut bracket_count = 1;
    let mut index = opening_bracket_index + 1;

    while index < tokens.len() {
        if let Token::Bracket(c, _) = &tokens[index] {
            if *c == opening_bracket {
                bracket_count += 1;
            } else if *c == closing_bracket {
                bracket_count -= 1;
            }

            if bracket_count == 0 {
                return index;
            }
        }

        index += 1;
    }

    panic!("No matching closing bracket found");
}

fn process_tokens(tokens: &Vec<Token>) -> TokenTreeRec {
    let mut min_level: usize = 0;
    let mut max_level: usize = 0;
    for token in tokens {
        println!("{} -  {:?}", get_nest_level(token), token);
        if min_level == 0 || get_nest_level(token) < min_level {
            min_level = get_nest_level(token);
        }
        if max_level == 0 || get_nest_level(token) > max_level {
            max_level = get_nest_level(token);
        }
    }
    println!("levels: {} {}", min_level, max_level);
    let mut tree = TokenTreeRec::new(Token::Keyword("Code".to_string(), 0));
    for token in tokens {
        if get_nest_level(token) != max_level {
            tree.children.push(TokenTreeRec::new(token.clone()));
        } else {
            let index = tree.children.len() - 1;
            tree.children[index].children.push(TokenTreeRec::new(token.clone()));
        }
    }
    println!("{:#?}", tree);
    return tree;
}

fn process_tokens_tree(tree: TokenTreeRec) -> TokenTreeRec {
    let tree2 = tree.clone();
    let mut min_level: usize = 0;
    let mut max_level: usize = 0;
    for token in tree.children {
        println!("{} -  {:?}", get_nest_level(&token.token), token);
        if min_level == 0 || get_nest_level(&token.token) < min_level {
            min_level = get_nest_level(&token.token);
        }
        if max_level == 0 || get_nest_level(&token.token) > max_level {
            max_level = get_nest_level(&token.token);
        }
        // println!("{:#?}", TokenTree::new(token.clone()));
    }
    println!("levels: {} {}", min_level, max_level);

    let mut tree3 = TokenTreeRec::new(Token::Keyword("Code".to_string(), 0));

    for token in tree2.clone().children {
        if get_nest_level(&token.token) != max_level {
            tree3.children.push(TokenTreeRec::new(token.token));
        } else {
            let index = tree3.children.len() - 1;
            let mut newToken = TokenTreeRec::new(token.token);
            newToken.children = token.children;
            tree3.children[index].children
                // here we loose token.children
                .push(newToken);
        }
    }

    return tree3;
}

fn is_empty<T>(vector: &[T]) -> bool {
    vector.len() == 0
}

fn process_multiply_signs(tree: TokenTreeRec) -> (bool, TokenTreeRec) {
    let mut tree2 = TokenTreeRec::new(Token::Keyword(get_name(&tree.token), 0));
    let mut min_level: usize = 0;
    let mut max_level: usize = 0;
    let mut i = 0;
    let mut found = true;
    found = false;

    while i < tree.children.len() - 2 {
        if
            !found &&
            (get_name(&tree.clone().children[i + 1].token) == "*" ||
                get_name(&tree.clone().children[i + 1].token) == "/") &&
            is_empty(&tree.clone().children[i + 1].children)
        {
            // current item is elem1
            // next item is '*'
            // after it is elem2
            let mut newToken = tree.children[i + 1].clone();
            newToken.children.push(tree.children[i].clone());
            newToken.children.push(tree.children[i + 2].clone());
            tree2.children.push(newToken);
            i += 2;
            found = true;
            println!("found);");
        } else {
            let mut newToken = tree.children[i].clone();
            tree2.children.push(newToken);
        }
        i += 1;
    }
    while i < tree.children.len() {
        let mut newToken = tree.children[i].clone();

        if newToken.children.len() >= 3 {
            let (_, processed) = process_multiply_signs(newToken.clone());
            tree2.children.push(processed);
            // tree2.children.push(newToken);
        } else {
            tree2.children.push(newToken);
        }
        i += 1;
    }

    (found, tree2)
}

fn process_sum_signs(tree: TokenTreeRec, lookupOperators: Vec<String>) -> (bool, TokenTreeRec) {
    let mut tree2 = TokenTreeRec::new(Token::Keyword(get_name(&tree.token), 0));
    let mut min_level: usize = 0;
    let mut max_level: usize = 0;
    let mut i = 0;
    let mut found = true;
    found = false;

    while i < tree.children.len() - 2 {
        if
            !found &&
            lookupOperators.contains(&&get_name(&tree.clone().children[i + 1].token).to_string()) &&
            is_empty(&tree.clone().children[i + 1].children)
        {
            // current item is elem1
            // next item is '*'
            // after it is elem2
            let mut newToken = tree.children[i + 1].clone();
            newToken.children.push(tree.children[i].clone());
            newToken.children.push(tree.children[i + 2].clone());
            tree2.children.push(newToken);
            i += 2;
            found = true;
            println!("found);");
        } else {
            let mut newToken = tree.children[i].clone();
            tree2.children.push(newToken);
        }
        i += 1;
    }
    while i < tree.children.len() {
        let mut newToken = tree.children[i].clone();

        if newToken.children.len() >= 3 {
            let (_, processed) = process_multiply_signs(newToken.clone());
            tree2.children.push(processed);
            // tree2.children.push(newToken);
        } else {
            tree2.children.push(newToken);
        }
        i += 1;
    }

    (found, tree2)
}

fn findUngroupedOperator(tree: TokenTreeRec, operator: Vec<String>) -> Vec<usize> {
    let mut path = Vec::new();
    let mut i = 0;
    while i < tree.children.len() {
        if
            operator.contains(&get_name(&tree.children[i].token)) &&
            tree.children[i].children.len() == 0
        {
            // found DIRECT uncollapsed operator
            path.push(i);
            return path;
        }
        i += 1;
    }
    // has no DIRECT uncollapsed operator
    i = 0;
    while i < tree.children.len() {
        if tree.children[i].children.len() > 0 {
            let mut innerPath = findUngroupedOperator(tree.children[i].clone(), operator.clone());
            if innerPath.len() > 0 {
                innerPath.insert(0, i);
                return innerPath;
            }
        }
        i += 1;
    }
    path
}

fn hasLevel(tree: TokenTreeRec, level: usize) -> bool {
    let mut i = 0;
    for token in tree.children {
        if level == get_nest_level(&token.token) {
            return true;
        }
        // println!("{:#?}", TokenTree::new(token.clone()));
    }
    return false;
}

// fn getByPath(tree:TokenTreeRec, path:Vec<usize>) -> *mut TokenTreeRec {
//     let mut res = &tree;
//     for i in path{
//         res = &res.children[i];
//     }
//     res

// }

fn getByPath(tree: &mut TokenTreeRec, path: Vec<usize>) -> &mut TokenTreeRec {
    let mut res = tree;
    for i in path {
        res = &mut res.children[i];
    }
    res
}

fn setByPath(tree: &mut TokenTreeRec, path: Vec<usize>, newTree: TokenTreeRec) {
    let mut tree2 = tree;

    for i in &path {
        tree2 = &mut tree2.children[*i];
    }

    *tree2 = newTree;
}

fn nestAdjascents(tree: TokenTreeRec) -> (bool, TokenTreeRec) {
    let mut tree2 = TokenTreeRec::new(Token::Keyword(get_name(&tree.token), 0));
    let mut found = true;
    found = false;

    let mut i = 0;

    while i < tree.children.len() - 2 {
        if
            vec!["if".to_string(), "while".to_string(), "func".to_string()].contains(
                &get_name(&tree.clone().children[i].token)
            )
        {
            // current item is if
            // next item is '*'
            // after it is elem2
            let mut newToken = tree.children[i].clone();
            newToken.children.push(tree.children[i + 1].clone());
            let mut body = tree.children[i + 2].clone();
            if body.children.len() >= 2 {
                let (_, nestedBody) = nestAdjascents(body.clone());
                let (_, nestedBody) = nestCalls(nestedBody.clone());
                newToken.children.push(nestedBody.clone());
            } else {
                newToken.children.push(body.clone());
            }

            tree2.children.push(newToken);
            i += 2;
            found = true;
            println!("found);");
        } else {
            if(vec!["return".to_string()].contains(
                &get_name(&tree.clone().children[i].token)
            )){

                let mut newToken = tree.children[i].clone();
                newToken.children.push(tree.children[i + 1].clone());
                tree2.children.push(newToken);
                i += 1;
            }
            else{
                let mut newToken = tree.children[i].clone();
                tree2.children.push(newToken);
            }
        }
        
        i += 1;
    }
    while i < tree.children.len() {




        if(vec!["return".to_string()].contains(
            &get_name(&tree.clone().children[i].token)
        )){

            let mut newToken = tree.children[i].clone();
            newToken.children.push(tree.children[i + 1].clone());
            tree2.children.push(newToken);
            i += 1;
        }
        else{
            let mut newToken = tree.children[i].clone();
            tree2.children.push(newToken);
        }



        let mut newToken = tree.children[i].clone();

        // if(newToken.children.len() >= 3){
        //     let (_, processed) =  nestAdjascents(newToken.clone());
        //     tree2.children.push(
        //         processed
        //     );
        // }
        // else{
        // tree2.children.push(newToken);
        // }
        i += 1;
    }

    (found, tree2)
}


fn nestCalls(tree: TokenTreeRec) -> (bool, TokenTreeRec) {
    let mut tree2 = TokenTreeRec::new(Token::Keyword(get_name(&tree.token), 0));
    let mut found = true;
    found = false;

    let mut i = 0;

    while i < tree.children.len() - 1 {
        if
            (vec!["Name".to_string()].contains( // TODO: real function name check
                &get_type(&tree.clone().children[i].token)
            ) && get_name(&tree.clone().children[i+1].token)=="(" ) 
        {
            // current item is if
            // next item is '*'
            // after it is elem2
            let mut newToken = tree.children[i].clone();
            let (_, mut args) =  nestCalls(tree.children[i + 1].clone()); 
            // newToken.children.push(args);

            // no body in call

            // let mut body = tree.children[i + 2].clone();
            // if body.children.len() >= 3 {
            //     let (_, nestedBody) = nestAdjascents(body.clone());
            //     newToken.children.push(nestedBody.clone());
            // } else {
            //     newToken.children.push(body.clone());
            // }

            newToken.children=vec![];

            let mut callToken =  TokenTreeRec{
                token:(Token::Keyword("FunctionCall".to_string(), 0)),
                children:vec![
                    newToken,
                ]
            };

            for arg in args.children{
                callToken.children.push(arg);
            }
            // callToken.children.push(args);

            tree2.children.push(callToken);//newToken);
            i += 1;
            found = true;
            println!("found);");
        } else {
            let mut newToken = tree.children[i].clone();
            tree2.children.push(newToken);
        }
        i += 1;
    }
    while i < tree.children.len() {
        let mut newToken = tree.children[i].clone();

        // if(newToken.children.len() >= 3){
        //     let (_, processed) =  nestAdjascents(newToken.clone());
        //     tree2.children.push(
        //         processed
        //     );
        // }
        // else{
        tree2.children.push(newToken);
        // }
        i += 1;
    }

    (found, tree2)
}

#[derive(Clone, Copy, Debug)]
enum SimDataType {
    Float,
    Vector,
    Object,
    Bool,
    Error,
    Null
}

// #[derive(Clone)]
union SimDataPayload{
    fValue: f32,
    bValue: bool,
    vValue: ManuallyDrop<Box<Vec<SimData>>>
}


impl Clone for SimDataPayload {
    fn clone(&self) -> Self {
        unsafe {
            match self {
                Self { fValue } => Self { fValue: *fValue },
                Self { bValue } => Self { bValue: *bValue },
                Self { vValue } => Self {
                    vValue: ManuallyDrop::new((**vValue).clone()),
                },
            }
        }
    }
}


impl fmt::Debug for SimDataPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            match self {
                SimDataPayload { fValue } => write!(f, "fValue: {}", fValue),
                SimDataPayload { bValue } => write!(f, "bValue: {}", bValue),
            }
        }
    }
}


// #[derive(Clone, Debug)]
// struct SimData{
//     dataType: SimDataType,
//     data: SimDataPayload,
// }

#[derive(Clone, Debug)]
enum SimData {
    Null,
    Bool(bool),
    Float(f64),
    Vector(Vec<SimData>),
    String(String),
    Object(HashMap<String, SimData>),
    Error(String),
}


impl Drop for SimData {
    fn drop(&mut self) {
        // TODO: implement
        // match &self {
        //     Self::Vector(v) =>{ 
        //         ManuallyDrop::drop(&mut ManuallyDrop::new(v));
        //     }
        // }
        // if let SimDataType::Vector = self.dataType {
        //     unsafe {
        //         ManuallyDrop::drop(&mut self.data.vValue);
        //     }
        // }
    }
}

impl fmt::Display for SimData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimData::Float(v) => write!(f, "{}", v ),
            SimData::Bool(v) => write!(f, "{}", v ),
            SimData::Null => write!(f, "Null",),
            _ => write!(f, "unsupported type"),
        }
    }
}

impl SimData{

    // initialization

    fn createFloat(v:f64) -> SimData{
        SimData::Float(v)
    }

    fn createBool(v:bool) -> SimData{
        SimData::Bool(v)
    }

    fn createNull() -> SimData {
        SimData::Float(0.0)
    }

    fn createVector(v: Vec<SimData>) -> SimData {
        SimData::Vector(ManuallyDrop::new(Box::new(v)).to_vec())
    }

    fn createString(v: String) -> SimData {
        SimData::String(v)
    }

    fn createObject(map: HashMap<String, SimData>) -> SimData{
        SimData::Object(map)
    }

    //misc

    fn dataTypeName(self) -> String {
        match self {
            SimData::Float(ref v) => "Float".to_string(),
            SimData::Vector(ref v) => "Vector".to_string(),
            SimData::Object(ref v) => "Object".to_string(),
            SimData::Bool(_) => "Bool".to_string(),
            SimData::Error(_) => "Error".to_string(),
            SimData::String(_) => "String".to_string(),
            SimData::Null => "Null".to_string()
        }
    }

    // math
    fn sum(v1:SimData, v2:SimData) -> SimData{
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createFloat(v1+v2),
            _ => {
                println!("cannot add different data types");    
                process::exit(1);
            }
        }
    }

    fn sub(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createFloat(v1-v2),
            _ => {
                println!("cannot subsract different data types");    
                process::exit(1);
            }
        }
    }

    fn mul(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createFloat(v1*v2),
            _ => {
                println!("cannot multiply different data types");    
                process::exit(1);
            }
        }
    }

    fn div(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createFloat(v1/v2),
            _ => {
                println!("cannot divide different data types");    
                process::exit(1);
            }
        }
    }

    // logical operators

    fn and(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Bool(v1), SimData::Bool(v2)) => return SimData::createBool(v1 && v2),
            _ => {
                println!("cannot use 'and' for non-bool data");    
                process::exit(1);
            }
        }
    }

    fn or(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Bool(v1), SimData::Bool(v2)) => return SimData::createBool(v1 || v2),
            _ => {
                println!("cannot use 'or' for non-bool data");    
                process::exit(1);
            }
        }
    }

    fn nor(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Bool(v1), SimData::Bool(v2)) => return SimData::createBool(!(v1||v2)),
            _ => {
                println!("cannot use 'nor' for non-bool data");    
                process::exit(1);
            }
        }
    }
    
    fn xor(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Bool(v1), SimData::Bool(v2)) => return SimData::createBool(!(v1==v2)),
            _ => {
                println!("cannot use 'xor' for non-bool data");    
                process::exit(1);
            }
        }
    }
    
    fn not(v: SimData) -> SimData {
        match v{
            SimData::Bool(v1) => return SimData::createBool(!v1),
            _ => {
                println!("cannot use 'not' for non-bool data");    
                process::exit(1);
            }
        }
    }

    // comparisons

    fn gt(v1: SimData, v2: SimData) -> SimData {
        println!("{:?} {:?}",v1, v2);
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createBool(v1>v2),
            _ => {
                println!("cannot use 'gt' for non-float data");    
                process::exit(1);
            }
        }
    }
    
    fn gte(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createBool(v1>=v2),
            _ => {
                println!("cannot use 'gte' for non-float data");    
                process::exit(1);
            }
        }
    }
    
    fn lt(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createBool(v1<v2),
            _ => {
                println!("cannot use 'lt' for non-float data");    
                process::exit(1);
            }
        }
    }
    
    fn lte(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createBool(v1<=v2),
            _ => {
                println!("cannot use 'lte' for non-float data");    
                process::exit(1);
            }
        }
    }
    
    fn eq(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createBool(v1==v2),
            _ => {
                println!("cannot use 'eq' for non-float data");    
                process::exit(1);
            }
        }
    }
    
    fn neq(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createBool(!(v1==v2)),
            _ => {
                println!("cannot use 'gt' for non-float data");    
                process::exit(1);
            }
        }
    }

    // convertors

    fn readFloat(self) -> f64 {
        match self{ 
            SimData::Float(v) => {return v;}
            _ => {return 0.0;}
        }
    }

    fn readBool(self) -> bool {
        match self{ 
            SimData::Bool(v) => {return v}
            _ => {return false}
        }
    }

    fn readVector(&self) -> Vec<SimData> {
        match self{ 
            SimData::Vector(v) => {return v.clone();}
            _ => {
                println!("Cannot read non-vector as a vector");
                let vRes = vec![SimData::Float(0.0)];
                return vRes;
            }
        }
    }

    fn readString(&self) -> String {
        match self{ 
            SimData::String(v) => {return v.clone()}
            _ => {
                println!("Cannot read non-vector as a vector");
                let sRes = "".to_string();
                return sRes;
            }
        }
    }

    fn readObject(&self) -> HashMap<String, SimData>{
        match self{ 
            SimData::Object(v) => {return v.clone()}
            _ => {
                println!("Cannot read non-object as a object");
                return HashMap::new();
            }
        }
    }

    // vector utilities

    fn setValueByIndex(&mut self, index:usize, value:Self) {
        match self{ 
            SimData::Vector(ref mut v) => { 
                v[index]=value;
            }
            _ => {
                println!("Cannot write by index to non-vector value");
            }
        }
    }


    fn push(&mut self, value:Self) {
        match self{ 
            SimData::Vector(ref mut v) => { 
                v.push(value);
            }
            _ => {
                println!("Cannot write by index to non-vector value");
            }
        }
    }
}



#[derive(Clone)]
struct ContextScope {
    context: Rc<RefCell<HashMap<String, SimData>>>,
    parent_scope: Option<Rc<ContextScope>>,
}

impl ContextScope {
    fn new() -> ContextScope {
        ContextScope {
            context: Rc::new(RefCell::new(HashMap::new())),
            parent_scope: None,
        }
    }

    fn extend(&self) -> ContextScope {
        ContextScope {
            context: Rc::new(RefCell::new(HashMap::new())),
            parent_scope: Some(Rc::new(self.clone())),
        }
    }

    fn set(&self, name: String, value: SimData) {
        let mut current_scope = self;
        // while let Some(scope) = &current_scope.parent_scope {
        //     if let Some(var) = scope.context.borrow().get(&name) {
        //         scope.context.borrow_mut().insert(name.clone(), value);
        //         return;
        //     }
        //     current_scope = scope;
        // }
        // if let Some(parent_scope) = &self.parent_scope {
        //     if parent_scope.get(name.clone()).is_some() {
        //         parent_scope.set(name.clone(), value.clone());
        //     }
        // }
        
        // let context = self.context.borrow();
        // if let Some(_) = context.get(&name) {
            self.context.borrow_mut().insert(name, value);
        // }
        // self.context.borrow_mut().insert(name, value);
    }

    fn pset(&self, name: String, value: SimData){

        if let Some(parent_scope) = &self.parent_scope {
            if parent_scope.has(name.clone()){
                parent_scope.pset(name.clone(), value)
            }
            else {
                self.set(name, value)
            } 
        } else {
            self.set(name, value)
        }
    }

    fn has(&self, name: String) -> bool {
        
        let context_ref = self.context.borrow();
        if let Some(value) = context_ref.get(&name) {
            true
        }else{false}
    }

    fn get(&self, name: String) -> SimData {
        if let Some(value) = self.context.borrow().get(&name) {
            return value.clone();
        }

        if let Some(parent_scope) = &self.parent_scope {
            return parent_scope.get(name);
        }

        SimData::createNull()
        // let context_ref = self.context.borrow();
        // if let Some(value) = context_ref.get(&name) {
        //     *value
        // } else if let Some(parent_scope) = &self.parent_scope {
        //     parent_scope.get(name)
        // } else {
        //     panic!("Variable not found: {}", name);
        // }
    }
}


/// new
#[derive(Clone, Debug)]
struct Context {
    values: HashMap<String, i32>,
    parent: Option<Box<Context>>,
}

impl Context {
    fn new() -> Self {
        Context {
            values: HashMap::new(),
            parent: None,
        }
    }

    fn extend(&self) -> Self {
        Context {
            values: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    fn get(&self, name: &str) -> Option<i32> {
        match self.values.get(name) {
            Some(value) => Some(*value),
            None => {
                if let Some(parent) = &self.parent {
                    parent.get(name)
                } else {
                    None
                }
            }
        }
    }

    fn set(&mut self, name: String, value: i32) {
        if let Some(parent) = &mut self.parent {
            if parent.get(&name).is_some() {
                parent.set(name.clone(), value);
                return;
            }
        }
        self.values.insert(name, value);
    }
}


unsafe fn test_vec_read(vec_value: Option<SimData>) -> Option<Vec<SimData>>{
    match vec_value{
        Some(sd) => { return Some(sd.readVector().to_vec()); }
        None => { return None }
    }
    // match vec_value{

    //     Some(SimData::Vector(v)) => { //{dataType,data}
    //         match data{
    //             SimDataPayload { vValue } =>{
    //                 return Some((&vValue).to_vec());
    //             }
    //         }
    //     }
    //     Some(_) => { return None }
    // }
}

fn testContext(){


    println!("================================================================");
    println!("= Testing ContextScope struct                                  =");
    println!("================================================================");

    let mut parent = ContextScope::new();
    parent.set(String::from("x".to_string()), SimData::createFloat(1.0));

    println!("Before set in parent (expect 1): parent.x={}",parent.get("x".to_string()));

    let mut child = parent.extend();
    
    println!("Before set in child (expect 1): child.x={}",child.get("x".to_string()));

    child.pset(
        "x".to_string(),
        SimData::sum( //7
            child.get("x".to_string()), //1
            SimData::createFloat(6.0)  //6
        )
    );

    child.pset(
        "y".to_string(),
        SimData::createFloat(8.0)  //8
    );


    println!("After set in child (expect 7): child.x={}",child.get("x".to_string()));
    println!("After set in child (expect 8): child.y={}",child.get("y".to_string()));
    println!("After set in parent (expect 7): parent.x={}",parent.get("x".to_string()));
    println!("After set in parent (expect Null): parent.y={}",parent.get("y".to_string())); 

    let mut root = ContextScope::new();
    root.set("a".to_string(), SimData::createFloat(2.0));

    let mut child1 = root.extend();
    child1.set("b".to_string(), SimData::createFloat(4.0));
    child1.pset("a".to_string(), SimData::sum(child1.get("a".to_string()), SimData::createFloat(1.0))); // 2 + 1 = 3

    let mut child2 = child1.extend();
    child2.set("c".to_string(), SimData::createFloat(6.0));
    child2.pset("b".to_string(), SimData::sum(child2.get("b".to_string()), SimData::createFloat(2.0))); // 4 + 2 = 6

    let mut child3 = child2.extend();
    child3.set("d".to_string(), SimData::createFloat(8.0));
    child3.pset("c".to_string(), SimData::sum(child3.get("c".to_string()), SimData::createFloat(3.0))); // 6 + 3 = 9

    // Test get, set, pset
    assert_eq!(root.get("a".to_string()).readFloat(), 3.0);
    assert_eq!(child1.get("a".to_string()).readFloat(), 3.0);
    assert_eq!(child1.get("b".to_string()).readFloat(), 6.0);
    assert_eq!(child2.get("a".to_string()).readFloat(), 3.0);
    assert_eq!(child2.get("b".to_string()).readFloat(), 6.0);
    assert_eq!(child2.get("c".to_string()).readFloat(), 9.0);
    assert_eq!(child3.get("a".to_string()).readFloat(), 3.0);
    assert_eq!(child3.get("b".to_string()).readFloat(), 6.0);
    assert_eq!(child3.get("c".to_string()).readFloat(), 9.0);
    assert_eq!(child3.get("d".to_string()).readFloat(), 8.0);

    // Test createFloat and sum
    let f1 = SimData::createFloat(5.0);
    let f2 = SimData::createFloat(10.0);
    let sum = SimData::sum(f1, f2);
    assert_eq!(sum.readFloat(), 15.0);

    let mut parent = ContextScope::new();
    parent.set(String::from("x"), SimData::createFloat(1.0));
    parent.set(String::from("a"), SimData::createBool(true));
    
    let mut child = parent.extend();

    // Math operations
    child.pset(
        String::from("y"),
        SimData::sum( 
            child.get(String::from("x")),
            SimData::createFloat(6.0)
        )
    );

    child.pset(
        String::from("z"),
        SimData::mul(
            child.get(String::from("y")),
            SimData::createFloat(2.0)
        )
    );

    child.pset(
        String::from("w"),
        SimData::div(
            child.get(String::from("z")),
            SimData::createFloat(4.0)
        )
    );

    // Logical operations
    child.pset(
        String::from("b"),
        SimData::and(
            child.get(String::from("a")),
            SimData::createBool(false)
        )
    );

    child.pset(
        String::from("c"),
        SimData::or(
            child.get(String::from("a")),
            SimData::createBool(false)
        )
    );

    child.pset(
        String::from("d"),
        SimData::not(
            child.get(String::from("a"))
        )
    );


    child.pset(
        String::from("e"),
        SimData::gt(
            child.get( String::from("y") ), 
            child.get( String::from("z") )
        )
    );

    child.pset(
        String::from("s1"),
        SimData::String(
            String::from("Hello world")
        )
    );


    println!("\nMath operations:");
    println!("y = x + 6: child.y={} (7)", child.get(String::from("y")).readFloat()); // 7
    println!("z = y * 2: child.z={} (14)", child.get(String::from("z")).readFloat()); // 14
    println!("w = z / 4: child.w={} (3.5)", child.get(String::from("w")).readFloat()); // 3.5

    println!("\nLogical operations:");
    println!("b = a AND false: child.b={} (false)", child.get(String::from("b")).readBool()); // false
    println!("c = a OR false: child.c={} (true)", child.get(String::from("c")).readBool()); // true
    println!("d = NOT a: child.d={} (false)", child.get(String::from("d")).readBool()); // false

    println!("\nComparison operations:");
    println!("e = y > z: child.e={} (false)", child.get(String::from("e")).readBool()); // 3.5


    println!("\nStrings:");
    println!("s1 = {} (\"Hello world\")", child.get(String::from("s1")).readString()); // 3.5


    println!("\nVectors:");

    let mut vInside = SimData::createVector(
        vec![
            SimData::createFloat(3.0),
            SimData::createFloat(8.0),
        ]
    );
    
    child.pset(
        String::from("v"),
        SimData::createVector(
            vec![
                SimData::createFloat(6.0),
                SimData::createFloat(61.0),
                vInside,
                SimData::createFloat(62.0),
                SimData::createVector(vec![
                    SimData::createVector(vec![
                        SimData::createVector(vec![
                            SimData::createFloat(2007.0),
                            child.get(String::from("s1")),
                        ])
                    ])  
                ])
            ]
        )
    );

    child.pset(
        String::from("o"), 
        SimData::Object(HashMap::from([
            (String::from("c"), SimData::Float(10.0)),
            (String::from("d"), SimData::Float(1.0)),
        ]))
    );

    let vec_value = child.context.borrow().get("v").cloned();
    let obj_value = child.context.borrow().get("o").cloned();


    if let Some(mut v) = vec_value{
        let mut vInner = &v.readVector().to_vec()[2];
        let mut vInner0 = &vInner.readVector().to_vec()[0];
        println!("v [2][0]: {} (3)", vInner0);   
        let mut vInner1 = &vInner.readVector().to_vec()[1];
        println!("v [2][1]: {} (8)", vInner1);   
        println!("v [3]: {} (62)", &v.readVector().to_vec()[3].clone().readFloat());    
        println!("v [1]: {} (61)", &v.readVector().to_vec()[1].clone().readFloat());    
        println!("v [0]: {} (6)", &v.readVector().to_vec()[0].clone().readFloat());   
        println!("v [4][0][0][0]: {} (2007)", &v.readVector().to_vec()[4].clone().readVector().to_vec()[0].clone().readVector().to_vec()[0].clone().readVector().to_vec()[0].clone().readFloat());   
        println!("v [4][0][0][1]: {} (\"Hello world\")", &v.readVector().to_vec()[4].clone().readVector().to_vec()[0].clone().readVector().to_vec()[0].clone().readVector().to_vec()[1].clone().readString());   
        println!("v [4][0][0][1]: {} (\"Hello world\")", &v.readVector().to_vec()[4].clone().readVector().to_vec()[0].clone().readVector().to_vec()[0].clone().readVector().to_vec()[1].clone().readString());   
        &v.push(SimData::Float(100.0));   
        println!("v [5]: {} (100)", &v.readVector().to_vec()[5].clone().readFloat());    
        &v.setValueByIndex(5,SimData::createFloat(2007.0));   

        println!("v [5]: {} (2007)", &v.readVector().to_vec()[5].clone().readFloat());    
    }

    if let Some (mut o) = obj_value {

        println!("\nObjects: ");
        if let Some(v) = &o.readObject().get("c") {
            println!("o.c: {} (10)", v );    
        }
        // if let Some(v) =  {
            println!("o.d: {} (1)", &o.readObject()["d"] );    
        // }

    }
    
    println!("\nAll tests passed!");
    
}

fn evaluate_r_value(tokenTree:TokenTreeRec, context:&mut ContextScope) -> SimData{
    if let TokenTreeRec{token:ref tokenRight, children:children} = tokenTree{
        // println!("name is {} {}", get_name(tokenRight), get_type(tokenRight));
        match tokenRight {
            Token::Number(name, level) => {
                return SimData::Float(get_number(&tokenRight));
            },
            Token::Keyword(name, level) =>{
                if(name=="(") {
                    return evaluate_r_value(children.clone()[0].clone(), context)
                }
            },
            Token::MathSign(name, level) => {
                if *name == "+" {
                    return SimData::sum(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == "-" {
                    return SimData::sub(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == "*" {
                    return SimData::mul(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == "/" {
                    return SimData::div(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == ">" {
                    return SimData::gt(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == "<" {
                    return SimData::lt(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == ">=" {
                    return SimData::gte(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == "<=" {
                    return SimData::lte(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == "==" {
                    return SimData::eq(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == "!=" {
                    return SimData::neq(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
                else if *name == "." {
                    // println!("{:#?}", children);
                    let left = children[0].clone();
                    let right = children[1].clone();
                    // println!("left: {:#?}", left);
                    let leftObj = evaluate_r_value(left, context);
                    // println!("leftObj: {:#?}", leftObj);
                    // println!("right: {:#?}", right.token);
                    match right.token {
                        Token::Keyword(..) | Token::Bracket(..) => {
                            let name =get_name(&right.token);
                            if(name=="("){
                                // println!("bracket",);
                                let index = evaluate_r_value(right.children[0].clone(), context);
                                // println!("bracket: {:?}", index);
                                // println!("data by index is: {:?}", leftObj.readVector().to_vec()[index.readFloat().round() as usize]);
                                return  leftObj.readVector().to_vec()[index.readFloat().round() as usize].clone();
                            }
                            else{
                                panic!("invalid expression");
                            }
                        }
                        _ => {
                            // code to execute if token is not a keyword or bracket
                        }
                    }
                    if let Token::Keyword(name, _) = right.token {
                        if(name=="("){
                            // println!("bracket",);
                            let index = evaluate_r_value(right.children[0].clone(), context);
                            // println!("bracket: {:?}", index);
                            // println!("data by index is: {:?}", leftObj.readVector().to_vec()[index.readFloat().round() as usize]);
                            return leftObj.readVector().to_vec()[index.readFloat().round() as usize].clone();
                        }
                        else{
                            panic!("invalid expression");
                        }
                    }else if let Token::Name(name, _) = right.token {
                        println!("field name {:?}", name);
                        // println!("object {:?}", leftObj);
                        match (leftObj.clone(), name.as_str()) {
                            (data, "type") => {
                                print!("getting type of {:?} -> {:?}", data, SimData::String(data.clone().dataTypeName()));
                                return SimData::String(data.dataTypeName());
                            },
                            (SimData::Vector(ref v), "length") => {
                                return SimData::Float(v.len() as f64);
                            },
                            (SimData::Object(ref v), _) => {
                                if let Some(data) = v.get(name.as_str()){
                                    return data.clone();
                                }
                                return SimData::Null;
                            },
                            _ => {}
                        }
                    }
                    panic!("Item Access - Not implemented!");
                    return SimData::neq(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
                }
            },
            Token::SpecialSign(name, level) => {},
            Token::Bracket(name, level) => {
                if *name=='[' {
                    let mut items: Vec<SimData> = vec![];
                    for i in children {
                        items.push(evaluate_r_value(i, context))
                    }
                    return SimData::createVector(items);
                }
                else{
                    process::exit(12);
                }
            },
            Token::Name(ref name, level) => {

                if(context.has(get_name(&tokenRight))){
                    return context.get(get_name(&tokenRight))
                }
            },
        };
        // if get_type(&tokenRight) == "Number" {
        //     println!("    right is: {}", get_number(&tokenRight));
        //     return SimData::Float(get_number(&tokenRight));
        // }
        // if get_type(&tokenRight) == "MathSign" {
        //     if get_name(&tokenRight) == "+" {
        //         return SimData::sum(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
        //     }
        //     else if get_name(&tokenRight) == "-" {
        //         return SimData::sub(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
        //     }
        //     else if get_name(&tokenRight) == "*" {
        //         return SimData::mul(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
        //     }
        //     else if get_name(&tokenRight) == "/" {
        //         return SimData::div(evaluate_r_value(children.clone()[0].clone(), context), evaluate_r_value(children.clone()[1].clone(), context))
        //     }
        // }
        // else if get_type(&tokenRight) == "Name"{
        //     if(context.has(get_name(&tokenRight))){
        //         return context.get(get_name(&tokenRight))
        //     }
        // }
        return SimData::Null;
        
        // context.set(name, SimData::Float(value))
    }
    return SimData::Null
}

fn execute_tree(context:&mut ContextScope, tokenTreeRec:TokenTreeRec){
    
    for i in tokenTreeRec.clone().children {
        if let TokenTreeRec{ref token, ref children} = i {
            let name = get_name(&token);
            if(name=="="){
                if let TokenTreeRec{token:tokenLeft, children:leftChildren} = &children[0]{
                    if let Token::Name(name, _) = &tokenLeft {
                        let name = get_name(&tokenLeft);
                        let mut value;
                        value = evaluate_r_value(children[1].clone(), context);
                        context.set(name, value)
                    } else if let Token::MathSign(name, _) = &tokenLeft {
                        println!("Left token is math sign");
                        if name == "." {
                            // panic!("dot children 0 is: {:?}", children[0].children[1]);
                            println!("Sign is .");
                            let vector_name = get_name(&children[0].children[0].clone().token);
                            println!("vector name: {:?}", vector_name);
                            println!("vector data: {:?}", evaluate_r_value(children[0].children[0].clone(), context));
                            let index = evaluate_r_value(children[0].children[1].children[0].clone(), context);
                            println!("index: {:?}",  index);
                            let value = evaluate_r_value(children[1].clone(), context);
                            println!("new value: {:?}", value);
                            let oldVector = &mut context.get(vector_name.clone());
                            // oldVector.setValueByIndex(index., value);
                            oldVector.setValueByIndex(index.readFloat().round() as i64 as usize, value);
                            context.set(vector_name, oldVector.clone());
                            // context.set(name, value)
                            
                            // panic!("proper dynamic index access");
                        } else {
                            panic!("unsupported lvalue type {:#?}", children[0]);
                        }
                    }
                    else{
                        panic!("unsupported lvalue type {:#?}", children[0]);
                    }
                }
            } 
            else if (name=="if"){
                println!("");
                
                // condition
                println!("condition:");
                println!("{:?}",i.clone().children[0].children[0]);
                println!("valuated: ");
                let mut res = evaluate_r_value(i.children[0].children[0].clone(), context);
                println!("{:?}",res);
                println!("");

                // body
                println!("body:");
                println!("{:?}",i.clone().children[1]);
                if (res.readBool()) {
                    execute_tree(context, i.children[1].clone());
                }

                println!("");
                println!("");
                // panic!("condition under construction")
            }
            else if (name=="while"){
                println!("");
                
                // condition
                println!("condition:");
                println!("{:?}",i.clone().children[0].children[0]);
                println!("valuated: ");
                let mut res = evaluate_r_value(i.children[0].children[0].clone(), context);
                println!("{:?}",res);
                println!("");

                // body
                println!("body:");
                println!("{:?}",i.clone().children[1]);
                while (res.readBool()) {
                    execute_tree(context, i.children[1].clone());
                    println!("valuated: ");
                    res = evaluate_r_value(i.children[0].children[0].clone(), context);
                    println!("{:?}",res);
                    println!("");
    
                }

                println!("");
                println!("");
                // panic!("condition under construction")
            }
            else{
                println!("Unknown action: {}", name);
            }
        }
    }

}

fn testExecution(){
    

    println!("================================================================");
    println!("= Testing code execition                                       =");
    println!("================================================================");
    /*
    
    
        varA = 1.5 + 0.75 - 0.25
        varB = 4 * 9 / 9
        text = 6.79
        testz = 4.5 / 3 - 0.5
        testy = 5.25 - 1.5 * 2 - 9 * 7
        testk = 6.75 / 1.5 + 0.25
        terf = varB

        varA = 2.0
        varB = 3.0
        varC = 2 * (2 + 1.0)
        varD = varC / 2.0 - varA

        xA = (5 + 3) * 4 / 2
        xB = (6 - 2) * (7 + 1) / 4
        xC = 2 * (3 + 4) - 5 / 2
        xD = ((2 + 3) * 4 + 7) / 5
        x1E = 6 > 9 - (3 - 1)
        x2E = 6 >= 9 - 3
        x3E = 6 <= 9 - 3
        x4E = 6 < 9 - 3
        x5e = 6 == 9 - 3
        x6e = 6 != 9 - 3
    
        if( xA  > 10 + 5 ){
            asdasdsadas = 2 + 9
            if( xA  > 10 + 6 ){
                qqqqqqqqqqqqqqq = 2 + 9
            }
            if( xA  > 10 + 4 ){
                eeeeeeeeeeeeeee = 5/3
            }
        }
     */
    let code =r#"

        ar = [
            11 
            2 
            [
                3
                9
            ]
            [
                1
                [
                    2
                    [
                        3
                    ]
                    [
                        1
                    ]
                ]
            ]
            x6e
            varA
            [
                [
                    [
                        x5e
                    ]
                ]
            ]
        ]

        sorted = [3.14 2.718 1.618 1.732 0.577 2.303 0.693 1.414 1.732 0.618]
        sl = sorted.length
        i = 0
        sum = 0
        while (i<19){
            sum = sum + sorted.(i)
            i = i + 1
        }

        k = 0
        s = 0


        repeats = 0
        while (k < sorted.length){
            k = k + 1
            s = 0
            while (s < sorted.length-1){
                if (sorted.(s) > sorted.(s+1)) {
                    temp = sorted.(s)
                    sorted.(s) = sorted.(s+1)
                    sorted.(s+1) = temp
                }
                repeats = repeats+1
                s = s + 1
            }
        }


        br = []
        cr = [1]
        dr = [2+2]



        decount = 12
        while ( decount > 5 ) {
            decount = decount - 2
        }

        ka = ar.(0)
        kb = ar.(2-1)
        kc = ar.(ar.(2-1)).(0)

        fi = pair.first
        si = pair.second
        ti = pair.third
        length = pair.length
        size = pair.length.size
        sizeType = size.type
        arType = ar.type
        pairType = pair.type
    "#;

    let mut parent = ContextScope::new();
    parent.set(
        "pair".to_string(), 
        SimData::Object(HashMap::from([
            (String::from("first"), SimData::Float(10.0)),
            (String::from("second"), SimData::Float(1.0)),
            (
                String::from("length"), 
                SimData::Object(HashMap::from([
                    ( "size".to_string(), SimData::Float(90.0) )
                ]))
            ),
    ])));


    let mut lexer = Lexer::new(code);
    let tokens = lexer.tokenize();

    let mut tokenTreeRec = process_tokens(&tokens);

    while hasLevel(tokenTreeRec.clone(), 2) {
        tokenTreeRec = process_tokens_tree(tokenTreeRec);
        println!("1st step");
        println!("{:#?}", tokenTreeRec);
    }

    (_, tokenTreeRec) = nestAdjascents(tokenTreeRec.clone());
    (_, tokenTreeRec) = nestCalls(tokenTreeRec.clone());

    let s = vec![
        vec![".".to_string()],
        vec!["*".to_string(), "/".to_string()],
        vec!["+".to_string(), "-".to_string()],
        vec![">".to_string(), "<".to_string(), ">=".to_string(), "<=".to_string(), "==".to_string(), "!=".to_string()],
        vec!["=".to_string()]
    ];

    let signs = s.clone();

    for signGroup in signs {
        let mut found = true;

        // need to properly loop it and make recursive
        while found {
            (found, tokenTreeRec) = process_sum_signs(tokenTreeRec, signGroup.clone());
        }
    }

    let signs = s.clone();

    for signGroup in signs {
        let mut address = findUngroupedOperator(tokenTreeRec.clone(), signGroup.clone());
        while address.len() > 0 {
            address.remove(address.len() - 1);
            // println!("address: {:#?}",address);
            let k = &mut tokenTreeRec;
            let res = process_sum_signs(getByPath(k, address.clone()).clone(), signGroup.clone()).1;
            setByPath(k, address.clone(), res);
            address = findUngroupedOperator(tokenTreeRec.clone(), signGroup.clone());
        }
    }

    println!("ended");
    println!("{:#?}", tokenTreeRec);

    let mut currentContext = parent;

    execute_tree(&mut currentContext, tokenTreeRec);

    // if let ContextScope{ context{RefCell}, parent_scope } = currentContext{

    // }
    let context_ref = currentContext.context.borrow();
    
    println!();
    println!("============ Execution results: =============================");

    for (name, data) in context_ref.iter() {
        println!("{} -> {:?}", name, data);
    }
    // context_ref.get(k)


}

fn main() {
    // let input =
    //     r#"
    //     testV = 3
    //     testVa = 4
    //     testVas = 5 > 0

    //     fib = func(x){
    //         res = fib(x-1) + fib(x-2)
    //         if (x>0) {
    //             fib(x-1)
    //             fib(x-1)
    //             return res
    //         }
    //         return 1
    //     }

    //     print(testValue)
    //     print(fib(testValue.data.[0+7-u.user], 9))

    //     if(x>0){
    //         print(9)
    //         print(9)
    //     }

    //     user(0)
    //     user.name(0)
    //     tos = user.name.toString
    //     tos(0)
    // "#;


    // let mut lexer = Lexer::new(input);
    // let tokens = lexer.tokenize();

    // let mut tokenTreeRec = process_tokens(&tokens);

    // while hasLevel(tokenTreeRec.clone(), 2) {
    //     tokenTreeRec = process_tokens_tree(tokenTreeRec);
    //     println!("1st step");
    //     println!("{:#?}", tokenTreeRec);
    // }

    // (_, tokenTreeRec) = nestAdjascents(tokenTreeRec.clone());
    // (_, tokenTreeRec) = nestCalls(tokenTreeRec.clone());

    // let s = vec![
    //     vec![".".to_string()],
    //     vec!["*".to_string(), "/".to_string()],
    //     vec!["+".to_string(), "-".to_string()],
    //     vec![">".to_string(), "<".to_string()],
    //     vec!["=".to_string()]
    // ];

    // let signs = s.clone();

    // for signGroup in signs {
    //     let mut found = true;

    //     // need to properly loop it and make recursive
    //     while found {
    //         (found, tokenTreeRec) = process_sum_signs(tokenTreeRec, signGroup.clone());
    //     }
    // }

    // let signs = s.clone();

    // for signGroup in signs {
    //     let mut address = findUngroupedOperator(tokenTreeRec.clone(), signGroup.clone());
    //     while address.len() > 0 {
    //         address.remove(address.len() - 1);
    //         // println!("address: {:#?}",address);
    //         let k = &mut tokenTreeRec;
    //         let res = process_sum_signs(getByPath(k, address.clone()).clone(), signGroup.clone()).1;
    //         setByPath(k, address.clone(), res);
    //         address = findUngroupedOperator(tokenTreeRec.clone(), signGroup.clone());
    //     }
    // }

    // println!("ended");
    // println!("{:#?}", tokenTreeRec);


    // println!("================================================================");
    // println!("= Testing SimData struct                                       =");
    // println!("================================================================");


    testContext();


    testExecution();

}