use std::alloc::System;
use std::process;
use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;
use std::mem::ManuallyDrop;
#[macro_use]
extern crate log;
use log::{info, warn};
use simple_logger::SimpleLogger;

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
                'a'..='z' | 'A'..='Z' | '_' => {
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
                '+' | '-' | '*' | '/' | '=' | '>' | '<' | '!'  | ':' | '@' | '$' => {
                    let mut name = current_char.to_string();
                    self.advance();
                    if ((name=='<'.to_string()) || (name=='>'.to_string()) || (name=='='.to_string()) || (name=='!'.to_string()) || (name==':'.to_string())  ){
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
            if c.is_alphabetic() || c.is_digit(10) || c=='_' {
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
        // println!("{} -  {:?}", get_nest_level(token), token);
        if min_level == 0 || get_nest_level(token) < min_level {
            min_level = get_nest_level(token);
        }
        if max_level == 0 || get_nest_level(token) > max_level {
            max_level = get_nest_level(token);
        }
    }
    // println!("levels: {} {}", min_level, max_level);
    let mut tree = TokenTreeRec::new(Token::Keyword("Code".to_string(), 0));
    for token in tokens {
        if get_nest_level(token) != max_level {
            tree.children.push(TokenTreeRec::new(token.clone()));
        } else {
            let index = tree.children.len() - 1;
            tree.children[index].children.push(TokenTreeRec::new(token.clone()));
        }
    }
    // println!("{:#?}", tree);
    return tree;
}

fn process_tokens_tree(tree: TokenTreeRec) -> TokenTreeRec {
    let tree2 = tree.clone();
    let mut min_level: usize = 0;
    let mut max_level: usize = 0;
    for token in tree.children {
        // println!("{} -  {:?}", get_nest_level(&token.token), token);
        if min_level == 0 || get_nest_level(&token.token) < min_level {
            min_level = get_nest_level(&token.token);
        }
        if max_level == 0 || get_nest_level(&token.token) > max_level {
            max_level = get_nest_level(&token.token);
        }
        // println!("{:#?}", TokenTree::new(token.clone()));
    }
    // println!("levels: {} {}", min_level, max_level);

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
            // println!("found);");
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
            // println!("found);");
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


fn process_unary_signs(tree: TokenTreeRec, lookupOperators: Vec<String>) -> (bool, TokenTreeRec) {
    let mut tree2 = TokenTreeRec::new(Token::Keyword(get_name(&tree.token), 0));
    let mut min_level: usize = 0;
    let mut max_level: usize = 0;
    let mut i = 0;
    let mut found = true;
    found = false;

    while i < tree.children.len() - 1 {
        if
            !found &&
            lookupOperators.contains(&&get_name(&tree.clone().children[i].token).to_string()) &&
            is_empty(&tree.clone().children[i].children)
        {
            // current item is unary sign
            // next item is elem
            let mut newToken = tree.children[i].clone();
            newToken.children.push(tree.children[i + 1].clone());
            tree2.children.push(newToken);
            i += 1;
            // found = true;
            // println!("found);");
        } else {
            let mut newToken = tree.children[i].clone();
            tree2.children.push(newToken);
        }
        i += 1;
    }
    while i < tree.children.len() {
        let mut newToken = tree.children[i].clone();

        if newToken.children.len() >= 2 {
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
            vec!["if".to_string(), "while".to_string(), "func".to_string(), ].contains(
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
            // println!("found);");
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
    if tree.children.len()>0{
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
            // println!("found);");
        } else {
            let mut newToken = tree.children[i].clone();
            tree2.children.push(newToken);
        }
        i += 1;
    }
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
    Null,
    Function
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

// impl Clone for &mut ContextScope{

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
    Function(Vec<String>,Vec<TokenTreeRec>, Option<*mut ContextScope>), // args and body
    Link(String, Vec<SimData>, usize), // name ob object in context scope, path inside this object and id of scope
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
            SimData::Function(..) => "Function".to_string(),
            SimData::Link(_,_,_) => "Link".to_string(),
            SimData::Null => "Null".to_string()
        }
    }

    // math
    fn sum(v1:SimData, v2:SimData) -> SimData{
        match (&v1, &v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createFloat(v1+v2),
            _ => {
                println!("cannot add following data types: {}, {}", v1.clone(), v2.clone());    
                process::exit(1);
            }
        }
    }

    fn sub(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createFloat(v1-v2),
            _ => {
                println!("cannot subsract following data types");    
                process::exit(1);
            }
        }
    }

    fn mul(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createFloat(v1*v2),
            _ => {
                println!("cannot multiply following data types");    
                process::exit(1);
            }
        }
    }

    fn div(v1: SimData, v2: SimData) -> SimData {
        match (v1, v2){
            (SimData::Float(v1), SimData::Float(v2)) => return SimData::createFloat(v1/v2),
            _ => {
                println!("cannot divide following data types");    
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
        // println!("{:?} {:?}",v1, v2);
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

    fn set_by_path(&self, path: &[SimData], value: SimData) -> SimData {
        if path.is_empty() {
            // Path cannot be empty
            panic!("Path cannot be empty");
        }

        let mut current_obj = self.clone();
        let mut remaining_path = path.iter();

        // Skip the first element of the path
        // remaining_path.next();


        if path.len() == 1{
            match current_obj {

                SimData::Object(ref mut map) => {
                    let obj_name = path[0].clone();
                    let obj = self.clone();
                    println!("set key {:?} to value {}", obj_name.readString(), value);
                    *map.entry(obj_name.readString()).or_insert(SimData::Null) = value;
                    return SimData::Object(map.clone());
                },
                SimData::Vector(ref mut vec) => {
                    let obj_name = &path[0].clone();
                    let obj = self.clone();
                    println!("set key {:?} to value {}", obj_name.clone().readFloat(), value);
                    vec[obj_name.clone().readFloat().round() as usize] = value;
                    return SimData::Vector(vec.clone());
                },
                _ =>{
                    error!("Object is required");
                }
            }
        } else{
            match current_obj {

                SimData::Object(ref mut map) => {
                    let obj_name = path[0].clone();
                    let obj = self.clone();
                    println!("set key {:?} to value {}", obj_name.readString(), value);

                    if let Some(x) = map.get( &obj_name.readString()){
                        let inner = x.clone();
                        let mut inner_path = path.clone();
                        if let Some((first, rest)) = inner_path.split_first() {
                            inner_path = rest;
                            // println!("The first element is: {}", first);
                            // println!("The rest of the array is: {:?}", inner_path);
                            // panic!("inner path:{:#?}", inner_path);
                        }
                        // panic!("in if let path long in obj is : {:#?}\nmap is: {:#?}\nobj name is: {:#?}", inner_path, map, &obj_name.readString());
                        *map.entry(obj_name.readString()).or_insert(SimData::Null) = x.clone().set_by_path(inner_path, value);
                    }
                    
                    return SimData::Object(map.clone());
                },
                SimData::Vector(ref mut vec) => {
                    let obj_name = &path[0].clone();
                    let obj = self.clone();
                    println!("set key {:?} to value {}", obj_name.clone().readFloat(), value);
                    // vec[obj_name.clone().readFloat().round() as usize] = value;
                    // from  above

                    if let Some(x) = vec.get( obj_name.clone().readFloat().round() as usize ) {
                        let inner = x.clone();
                        let mut inner_path = path.clone();
                        if let Some((first, rest)) = inner_path.split_first() {
                            inner_path = rest;
                            // println!("The first element is: {}", first);
                            // println!("The rest of the array is: {:?}", inner_path);
                            // panic!("inner path:{:#?}", inner_path);
                        }
                        // panic!("in if let path long in obj is : {:#?}\nmap is: {:#?}\nobj name is: {:#?}", inner_path, map, &obj_name.readString());
                        vec[obj_name.clone().readFloat().round() as usize] = x.clone().set_by_path(inner_path, value);
                    }
                    
                    // end from above
                    return SimData::Vector(vec.clone());
                },
                _ =>{
                    error!("Object is required");
                }
            }
        }
        panic!("length more that 1 is not allowed");

        for obj_name in remaining_path {
            current_obj = match current_obj {
                SimData::Object(ref mut map) => {
                    let obj_name = obj_name.clone().to_string();
                    let obj = map.entry(obj_name).or_insert_with(|| SimData::Object(HashMap::new()));
                    obj.clone()
                },
                SimData::Vector(ref mut list) => {
                    let index = obj_name.clone().readFloat().round() as usize;
                    let obj = list.get_mut(index).expect("Index out of range");
                    obj.clone()
                },
                _ => panic!("Cannot get sub-object from non-composite type"),
            }
        }

        match current_obj {
            SimData::Object(ref mut map) => {
                let obj_name = path.last().unwrap().clone().to_string();
                map.insert(obj_name, value);
                SimData::Object(map.clone())
            },
            SimData::Vector(ref mut list) => {
                let index = path.last().unwrap().clone().readFloat().round() as usize;//.expect("Invalid index in path");
                list[index as usize] = value;
                SimData::Vector(list.clone())
            },
            _ => panic!("Cannot set value on non-composite type"),
        }
    }
}



#[derive(Clone, Debug)]
struct ContextScope {
    context: Rc<RefCell<HashMap<String, SimData>>>,
    links: RefCell<HashMap<String, String>>,
    parent_scope: Option<Rc<ContextScope>>,
    id: usize,
    count: Rc<RefCell<usize>>,
}

static mut current_context_id: usize = 0;

impl ContextScope {
    fn new() -> ContextScope {
        let count = Rc::new(RefCell::new(1));
        ContextScope {
            context: Rc::new(RefCell::new(HashMap::new())),
            links: RefCell::new(HashMap::new()),
            parent_scope: None,
            id: 1,
            count
        }
    }

    fn extend(&self) -> ContextScope {
        let count = self.count.clone();
        let new_id = {
            let mut count_ref = count.borrow_mut();
            *count_ref += 1;
            *count_ref
        };
        ContextScope {
            context: Rc::new(RefCell::new(HashMap::new())),
            links: RefCell::new(HashMap::new()),
            parent_scope: Some(Rc::new(self.clone())),
            id: new_id,
            count
        }
    }

    fn add_link(&self, local_name: String, parent_name: String) {
        self.links.borrow_mut().insert(local_name, parent_name);
    }

    fn set(&self, name: String, value: SimData) {
        if let Some(parent_name) = self.links.borrow().get(&name) {
            if let Some(parent_scope) = &self.parent_scope {
                parent_scope.set(parent_name.clone(), value);
            }
        } else {
            self.context.borrow_mut().insert(name, value);
        }
        // let mut current_scope = self;
        // self.context.borrow_mut().insert(name, value);
    }

    fn set_by_path(&self, path: Vec<SimData>, value: SimData) {
        if path.is_empty(){
            error!("Path cannot be empty");
        }
        let obj_name = path[0].to_string();
        let mut remaining_path = path;
        remaining_path.remove(0);
        let new_obj = self.get(obj_name);
        

        let mut i = 0;
        while i < remaining_path.len() - 1 {
            let obj_name = &remaining_path[i];
            
        }
        // iterate through path from 1st element (skip element with index 0) and get the value by it
    }

    fn pset(&self, name: String, value: SimData){
        if(self.has(name.clone())){
            return self.set(name, value);
        }
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
        
        if self.context.borrow().contains_key(&name) || self.links.borrow().contains_key(&name) {
            true
        } else {
            false
        }
    }

    fn get(&self, name: String) -> SimData {
        if let Some(value) = self.context.borrow().get(&name) {
            return value.clone();
        }

        if let Some(parent_name) = self.links.borrow().get(&name) {
            if let Some(parent_scope) = &self.parent_scope {
                return parent_scope.get(parent_name.clone());
            }
        }

        if let Some(parent_scope) = &self.parent_scope {
            return parent_scope.get(name);
        }

        SimData::createNull()
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




fn path_elem_from_token(tokenTreeRec:TokenTreeRec, context:&ContextScope) -> SimData {
    let TokenTreeRec{token, children} = tokenTreeRec;
    match token{
        Token::Name( name, _ ) => return SimData::String(name),
        Token::Bracket(_, _) => return evaluate_r_value(children[0].clone(), &mut context.clone()),
        _ => error!("Incorrect path token")
    }
    
    SimData::Null
}

fn get_path_from_dot_tree(tree: TokenTreeRec, context:&ContextScope) -> Vec<SimData> { // accepts TokenTreeRec with a dot as main element
    let mut path:Vec<SimData> = vec![];
    if get_name(&tree.children[0].token) == "."{ // first child is also a dot
        let mut old_path = get_path_from_dot_tree(tree.children[0].clone(), context);
        old_path.push(path_elem_from_token(tree.children[1].clone(), context));
        return old_path;
    } 
    // else
    return vec![path_elem_from_token(tree.children[0].clone(), context), path_elem_from_token(tree.children[1].clone(), context)]
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
    parent.set(String::from("data1".to_string()), SimData::createFloat(11.0));

    println!("Before set in parent (expect 1): parent.x={}",parent.get("x".to_string()));
    println!("Before set in parent (expect Null): parent.data1={}",parent.get("data1".to_string())); 

    let mut child = parent.extend();
    child.add_link("localData".to_string(), "data1".to_string());
    println!("Before set in child (expect 11): child.localData={}",child.get("localData".to_string())); 
    
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

    child.pset("localData".to_string(), SimData::Float(2007.0));



    println!("After set in child (expect 7): child.x={}",child.get("x".to_string()));
    println!("After set in child (expect 8): child.y={}",child.get("y".to_string()));
    println!("After set in parent (expect 7): parent.x={}",parent.get("x".to_string()));
    println!("After set in parent (expect Null): parent.y={}",parent.get("y".to_string())); 
    println!("After set in parent (expect 11): parent.data1={}",parent.get("data1".to_string())); 
    println!("After set in child (expect 11): child.localData={}",child.get("localData".to_string())); 

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

fn get_names(tokens: Vec<TokenTreeRec>) -> Vec<String> {
    let mut names = Vec::new();
    for token in tokens {
        match token.token {
            Token::Name(name, _) => names.push(name),
            _ => {
                // Recursively check children for names
                // names.append(&mut get_names(token.children));
            }
        }
    }
    names
}

fn execute_func_call(func_obj:SimData, args:Vec<SimData>, parentContext:&mut ContextScope) -> SimData {
    let mut context = parentContext.extend();
    if let SimData::Function(ref argNames, ref body, _) = func_obj {
        println!( "{:?} -> {:?}", argNames, args);
        if argNames.len() != args.len(){
            error!("expected {} arguments, but got {}", argNames.len(), args.len() )
        }
        for (i, argName) in argNames.iter().enumerate() {
            context.set(argName.to_string(), args[i].clone())
        }
        println!("body: {:?}", body[0]);
        return execute_tree(&mut context, TokenTreeRec{children:body.clone(), token:Token::Keyword("Code".to_string(), 1)});
    } else {
        error!("not callable");
    }
    SimData::Null
}

fn evaluate_r_value(tokenTree:TokenTreeRec, context:&mut ContextScope) -> SimData{
    println!("token is: {:?}", tokenTree.clone());
    if let TokenTreeRec{token:ref tokenRight, children:children} = tokenTree.clone(){
        println!("name is {} {}", get_name(tokenRight), get_type(tokenRight));
        match tokenRight {
            Token::Number(name, level) => {
                return SimData::Float(get_number(&tokenRight));
            },
            Token::Keyword(name, level) =>{
                if(name=="(") {
                    return evaluate_r_value(children.clone()[0].clone(), context)
                }

                else if *name=="[" {
                    // println!("parsing [ top");
                    let mut items: Vec<SimData> = vec![];
                    let mut obj:HashMap<String, SimData> = HashMap::new();
                    let mut mode  = 0; // vector
                    for i in children {
                        // println!("i = {:?}", i);
                        if let TokenTreeRec{ref token, ref children} = i {
                            if children.len() > 0 {
                            if let TokenTreeRec{ref token, children:childrenIn} = &children[0] {
                                // println!("\n left: {:?}", get_name(&token));
                                let tokenRight = children[1].clone();
                                // println!(" right: {:?}", tokenRight);
                                // println!(" right: {}", evaluate_r_value(tokenRight, context));
                                // println!(" {} -> {}", get_name(&token), evaluate_r_value(tokenRight, context));
                                mode = 1;
                                obj.insert(get_name(&token), evaluate_r_value(tokenRight, context));
                                // println!(" token obj?: {:?}, \n children: {:?} \n\n", token, children)å;
                                
                            }}
                        }
                        items.push(evaluate_r_value(i, context))
                    }
                    if mode == 1{
                        return SimData::createObject(obj);
                    }
                    return SimData::createVector(items);
                }

                else if *name=="func" {
                    println!("func found {:#?}", children);

                    println!("args: {:#?}", get_names(children[0].children.clone()) );
                    println!("body: {:#?}", children[1].children );
                    // here something is wrong
                    return SimData::Function(get_names(children[0].children.clone()) ,children[1].children.clone(), Some(context) );
                    process::exit(90);
                }

                // function call execution
                else if name=="FunctionCall"{
                    let functionTTR = children[0].clone();
                    let mut funcName="".to_string();
                    let mut argValues:Vec<SimData> = vec![];
                    println!("found function call :: {:?} {:?}", functionTTR, children);
                    if let TokenTreeRec { token:functionToken, children:functionChildren } = functionTTR.clone(){
                        // if let Token{name:functionName} = functionToken{
                            if let Token::Name(name, _) = functionToken{
                                funcName = name;
                            }
                            else{
                                panic!("calling")
                            }
                        // }
                    }
                    let mut i = 1;
                    while i < children.len(){
                        argValues.push(evaluate_r_value(children[i].clone(), context));
                        i += 1;
                    }
                    println!("function name: {:?}", funcName);
                    println!("function args: {:?}", argValues); 
                    println!("function obj: {:?}", context.get(funcName.clone()));
                    return execute_func_call(context.get(funcName.clone()).clone(), argValues, context);
                    // process::exit(12);
                }

                else{
                    error!("unknown rvalue type :: {:?}", tokenRight);
                    process::exit(12);
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
                else if *name == "@" {
                    if(get_name(&children.clone()[0].token)=="(".to_string()){
                        let TokenTreeRec{ token, children:c } = children[0].clone();
                        let full_path = get_path_from_dot_tree(children[0].children[0].clone(), context);
                        return SimData::Link(full_path[0].readString(), full_path[1..].to_vec(),context.id);
                        panic!("Today children of link operator are: {:#?}", get_path_from_dot_tree(children[0].children[0].clone(), context) );
                    } else {
                        let TokenTreeRec{ token, children:_ } = children[0].clone();
                        if let Token::Name( name, _) = token{
                            return SimData::Link(name.to_string(), vec![],context.id);
                            panic!("\n\nToday children of link operator are: {:#?}\n\n", vec![SimData::String(name)] );
                        }
                        else{
                            return SimData::Link("Hello".to_string(), vec![],context.id);
                            panic!("Incorrect call of link orperator: {:#?}", children.clone()[0] );
                        }
                    }
                    
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
                        // println!("field name {:?}", name);
                        // println!("object {:?}", leftObj);
                        match (leftObj.clone(), name.as_str()) {
                            (data, "type") => {
                                // print!("getting type of {:?} -> {:?}", data, SimData::String(data.clone().dataTypeName()));
                                return SimData::String(data.dataTypeName());
                            },
                            (SimData::Vector(ref v), "length") => {
                                return SimData::Float(v.len() as f64);
                            },
                            (SimData::Vector(ref v), "len") => {
                                return SimData::Function(vec![],vec![
                                    TokenTreeRec {
                                        token: Token::Keyword(
                                            "return".to_string(),
                                            2,
                                        ),
                                        children: vec![
                                            TokenTreeRec {
                                                token: Token::Bracket(
                                                    '(',
                                                    2,
                                                ),
                                                children: vec![
                                                    TokenTreeRec {
                                                        token: Token::Number(
                                                            v.len() as f64,
                                                            3,
                                                        ),
                                                        children: vec![],
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                ], None);
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
                    // println!("parsing [");
                    let mut items: Vec<SimData> = vec![];
                    let mut obj:HashMap<String, SimData> = HashMap::new();
                    for i in children {
                        if let TokenTreeRec{ref token, ref children} = i {
                            // println!("token obj?: {:?}", token);
                        }
                        items.push(evaluate_r_value(i, context))
                    }
                    return SimData::createVector(items);
                }
                else if *name=='(' {
                    // println!("parsing [");
                    // let mut items: Vec<SimData> = vec![];
                    // let mut obj:HashMap<String, SimData> = HashMap::new();
                    // for i in children {
                    //     if let TokenTreeRec{ref token, ref children} = i {
                    //         // println!("token obj?: {:?}", token);
                    //     }
                    //     items.push(evaluate_r_value(i, context))
                    // }
                    return evaluate_r_value(children[0].clone(), context);
                }
                else{
                    process::exit(12);
                }
            },
            Token::Name(ref name, level) => {

                if(context.has(get_name(&tokenRight))){
                    return context.get(get_name(&tokenRight))
                }
                warn!("Variable {} not defined and will be interpreted as Null", get_name(&tokenRight) );
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

fn execute_tree(context:&mut ContextScope, tokenTreeRec:TokenTreeRec) -> SimData{
    
    for i in tokenTreeRec.clone().children {
        if let TokenTreeRec{ref token, ref children} = i {
            let name = get_name(&token);
            if(name=="=") || (name==":=") {
                let morsh = name == ":=";
                // panic!("morsh {}",  morsh);
                // println!("evaluation equation {:?}  = {:?}", &children[0], &children[1]);
                if let TokenTreeRec{token:tokenLeft, children:leftChildren} = &children[0]{
                    if let Token::Name(name, _) = &tokenLeft {
                        let name = get_name(&tokenLeft);
                        let mut value;
                        println!("lookup right for left= {}",name);
                        value = evaluate_r_value(children[1].clone(), context);
                        if(morsh){
                            context.set(name, value)
                        }
                        else{
                            context.pset(name, value)
                        }
                    } else if let Token::MathSign(name, _) = &tokenLeft {
                        // println!("Left token is math sign");
                        if name == "." {
                            // panic!("dot children 0 is: {:?}", children[0].children[1]);
                            // println!("Sign is .");
                            let value = evaluate_r_value(children[1].clone(), context);
                            print!("\n\nDot found!\n\n");

                            let mut path = get_path_from_dot_tree(children[0].clone(), context);
                            let value = evaluate_r_value(children[1].clone(), context);
                            let (root, remaining_path) = path.split_first().expect("Path cannot be empty");
                            let old_obj = context.get(root.readString());
                            println!("\n\nname is  << {:?} >>", root);
                            println!("path is  << {:?} >>", remaining_path);
                            println!("value is  << {:?} >>", value);
                            println!("Old object is  << {:#?} >>\n\n", old_obj);
                            let new_obj = old_obj.set_by_path(remaining_path, value );
                            println!("\n\n new data: {:#?} \n\n", new_obj);
                            if(morsh){
                                context.set(root.readString(), new_obj);
                            }
                            else{
                                // panic!(":= morsh");
                                context.pset(root.readString(), new_obj);
                            }
                            continue;
                        } else if name == "$" {
                            let pointer = children[0].children[0].clone().token;
                            if let Token::Name(name, level) = pointer {
                                let mut pointer_data = context.get(name);
                                if let SimData::Link(name_in_ctx, path, level) = &pointer_data{
                                    println!("Name: {} \nPath:{:#?} \nLevel:{}", name_in_ctx, path, level);
                                    if(*level == 1 as usize){
                                        println!("set");
                                        let mut new_path = path.clone();
                                        new_path.insert(0, SimData::String(name_in_ctx.clone()));
                                        let old_obj = context.get(name_in_ctx.clone());
                                        let (root, remaining_path) = new_path.split_first().expect("Path cannot be empty");
                                        let value = evaluate_r_value(children[1].clone(), context);
                                        println!("path: {:?}", remaining_path.clone());
                                        let new_obj = old_obj.set_by_path(remaining_path, value );
                                        context.set(name_in_ctx.clone(), new_obj);
                                        // return SimData::Null; /
                                        // context.set_by_path(new_path.clone(), new_obj)
                                    }
                                    else{
                                        println!("level {} is not suppotred", level)
                                    }
                                }
                                // panic!("Asignment by pointer {:#?}", pointer_data);//children[0]);panic!("Asignment by pointer {:#?}", pointer);//children[0]);
                            } else if let Token::Bracket(name, level) = pointer {
                                panic!("Pointer path assignment {:#?}", children[0]);
                            } else if let Token::Keyword(name, level) = pointer {
                                if name=="(" {
                                    let path = get_path_from_dot_tree(children[0].children[0].children[0].clone(), context);
                                    let (root, remaining_path) = path.split_first().expect("Path cannot be empty");
                                    println!("\n\nroot: {:?}", root);
                                    println!("pointer: {:?}", context.get(root.readString()));
                                    println!("path is {:#?} \n\n", remaining_path);
                                    let mut d = & context.get(root.readString());
                                    if let SimData::Link(var_name, path, level) = d{
                                        println!("path is {:#?} \n\n", remaining_path);   
                                        let mut res_path:&mut Vec<SimData> = &mut path.clone();
                                        res_path.extend(remaining_path.to_vec().clone());
                                        let res_pointer = SimData::Link(var_name.clone(), res_path.clone(), level.clone());
                                        println!("pointer computed: {:#?} \n\n", res_pointer);

                                        let old_obj = context.get(var_name.clone());
                                        // let (root, remaining_path) = res_path.split_first().expect("Path cannot be empty");
                                        let value = evaluate_r_value(children[1].clone(), context);
                                        println!("path: {:?}", remaining_path.clone());
                                        let new_obj = old_obj.set_by_path(res_path, value );
                                        context.set(var_name.clone(), new_obj);
                                        // panic!("Pointer path assignment {:#?}", children[0].children[0].children[0]);
                                    } else {
                                        panic!("Pointer path assignment {:#?}", children[0].children[0].children[0]);
                                    }
                                }
                            } else {
                                panic!("Incorrect pointer assignment {:#?}", pointer);
                            }
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
                
                // condition
                // println!("condition:");
                // println!("{:?}",i.clone().children[0].children[0]);
                // println!("valuated: ");
                let mut res = evaluate_r_value(i.children[0].children[0].clone(), context);
                // println!("{:?}",res);
                // println!("");

                // body
                // println!("body:");
                println!("if cond:\n\n{:?}\n\n",i.clone().children[0]);
                // panic!("if  condition value:\n\n{}\n\n", res.readBool());
                if (res.readBool()) {
                    let exec_res = execute_tree(context, i.children[1].clone());
                    match exec_res{
                        SimData::Null =>{ },
                        _=>{
                            return exec_res;
                        }
                    }
                }

                // println!("");
                // println!("");
                // panic!("condition under construction")
            }
            else if (name=="while"){
                // println!("");
                
                // condition
                // println!("condition:");
                // println!("{:?}",i.clone().children[0].children[0]);
                // println!("valuated: ");
                let mut res = evaluate_r_value(i.children[0].children[0].clone(), context);
                // println!("{:?}",res);
                // println!("");

                // body
                // println!("body:");
                // println!("{:?}",i.clone().children[1]);
                while (res.readBool()) {

                    execute_tree(context, i.children[1].clone());
                    // println!("valuated: ");
                    res = evaluate_r_value(i.children[0].children[0].clone(), context);
                    // println!("{:?}",res);
                    // println!("");
    
                }

                // println!("");
                // println!("");
                // panic!("condition under construction")
            }
            else if (name=="return"){
                println!("returning: {:#?}", i);
                println!("return value: {:#?}", evaluate_r_value(i.children[0].clone(), context));
                return evaluate_r_value(i.children[0].clone(), context);
                error!("return not implemented");
            }
            else{
                error!("Unknown action: {}", name);
            }
        }
    }
    SimData::Null

}

fn testExecution(){
    

    println!("================================================================");
    println!("= Testing code execition                                       =");
    println!("================================================================");
    /*
    
        e = func(x y ){
            return x * 2
        }

        

        data = [    
            id = 1    
            sharedInterests = [ 5 6 9 ]   
        ]


        dataid = data.sharedInterests.length

        o = [ x  0 ]

        matrix = [
            row1 = [ 1 2 3 ]
            row2 = [ 4 5 6 ]
            row3 = [ 7 8 9 ]
            properties = [
                determinant = 0
                isSymmetric = [ 0 ]
                isInvertible = [ 0 ]
            ]
        ]

        settings = [
            window = [
                width = 800
                height = 600
                position = [ 100 200 ]
            ]
            theme = [
                primaryColor = [ 23 56 98 ]
                secondaryColor = [ 245 245 245 ]
            ]
            preferences = [
                notifications = [ 0 ]
                language = [ 69 110 103 108 105 115 104 ]
            ]
        ]

        data = [    
            id = 1   
            name = [ 74 111 104 110 ]
            age = 30
            scores = [ 100 90 80 ]
            friends = [        
                friend1 = [            
                    id = 2            
                    shared_interests = [ 3 5 7 ]
                ]
                friend2 = [            
                    id = 3            
                    shared_interests = [ 1 5 6 ]
                ]
            ]
        ]

        sorted = [3.14 2.718 1.618 1.732 0.577 100 2.303 2007 643 0.693 1.414 1.732 0.618]
        s = 0
        k = 0
        repeats = 0


        stackSize = sorted.length
        stack = [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
        

        top = 0-1
        
        top = top + 1
        stack.(top) = 0
        top = top + 1
        stack.(top) = sorted.length - 1
        
        while (top >= 0) {
            high = stack.(top)
            top = top - 1
            low = stack.(top)
            top = top - 1
        
            i = low - 1
            pivot = sorted.(high)
        
            j = low
        while (j <= high - 1) {
                if (sorted.(j) <= pivot) {
                    i = i + 1
        
                    temp = sorted.(i)
                    sorted.(i) = sorted.(j)
                    sorted.(j) = temp
                }
                j = j + 1
            }
        
            temp = sorted.(i + 1)
            sorted.(i + 1) = sorted.(high)
            sorted.(high) = temp
            pi = i + 1
        
            if (pi - 1 > low) {
                top = top + 1
                stack.(top) = low
                top = top + 1
                stack.(top) = pi - 1
            }
        
            if (pi + 1 < high) {
                top = top + 1
                stack.(top) = pi + 1
                top = top + 1
                stack.(top) = high
            }
        }
        

        q = 0
        d = 7 + q
        s = l
    
     */
    let code =r#"
    in1 = [ 1 2 3 ]
    contest = func (x){
        return (x+2)
    }
    
    contest2 = func (x){
        contest3 = func (y){
            return (x+y)
        }
        return contest3
    }
    

    d = contest2(3)


    "#;
    /*
    

        matrix = [
            row1 = [ 1 2 3 ]
            row2 = [ 4 5 6 ]
            row3 = [ 7 8 9 ]
            properties = [
                determinant = 0
                is_symmetric = [ 0 ]
                is_invertible = [ 0 ]
            ]
        ]

        settings = [
            window = [
                width = 800
                height = 600
                position = [ 100 200 ]
            ]
            theme = [
                primary_color = [ 23 56 98 ]
                secondary_color = [ 245 245 245 ]
            ]
            preferences = [
                notifications = [ 0 ]
                language = [ 69 110 103 108 105 115 104 ]
            ]
   
        ]
 */

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

    // print!("{:#?}", tokenTreeRec);
    // panic!("tree only mode");

    while hasLevel(tokenTreeRec.clone(), 2) {
        tokenTreeRec = process_tokens_tree(tokenTreeRec);
        // println!("1st step");
        // println!("{:#?}", tokenTreeRec);
    }

    (_, tokenTreeRec) = nestAdjascents(tokenTreeRec.clone());
    println!("Token tree generated");
    println!("{:#?}", tokenTreeRec);
    (_, tokenTreeRec) = nestCalls(tokenTreeRec.clone());

    (_,tokenTreeRec) = process_unary_signs(tokenTreeRec, vec!["@".to_string(),"$".to_string()]);

    let s = vec![
        vec![".".to_string(), ],
        vec!["*".to_string(), "/".to_string()],
        vec!["+".to_string(), "-".to_string()],
        vec![">".to_string(), "<".to_string(), ">=".to_string(), "<=".to_string(), "==".to_string(), "!=".to_string()],
        vec!["=".to_string(), ":=".to_string()] // ':=' is forced to local variable, '=' is looking up
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

    // println!("Token tree generated");
    // println!("{:#?}", tokenTreeRec);

    // process::exit(2007);

    let mut currentContext = parent;

    execute_tree(&mut currentContext, tokenTreeRec);

    // if let ContextScope{ context{RefCell}, parent_scope } = currentContext{

    // }
    let context_ref = currentContext.context.borrow();
    
    println!("============ Execution results: =============================");

    for (name, data) in context_ref.iter() {
        println!("{} -> {:#?}", name, data);
    }
    // context_ref.get(k)


}

fn main() {
    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug) // Set the minimum log level
        .init()
        .unwrap();

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