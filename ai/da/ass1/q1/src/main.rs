use std::vec::Vec;

fn dup(arr: Vec<u16>) -> bool {
    for i in 0..arr.len() {
        for j in 0..arr.len() {
            if i != j && arr[i] == arr[j] {
                return true;
            }
        }
    }

    false
}

fn rec(i: u16, mut arr: Vec<u16>, cap: &mut Vec<u16>) {
    let sum: u16 = arr.iter().sum();
    if sum == 15 && arr.len() == 3 && !dup(arr.clone()) {
        for i in arr {
            cap[(i - 1) as usize] += 1;
            print!("{i} ");
        }
        println!();

        return;
    }

    if arr.len() == 3 {
        return;
    }

    for j in i..=9 {
        arr.push(j);
        rec(j, arr.clone(), cap);
        arr.pop();
    }
}

fn main() {
    let mut arr = Vec::with_capacity(3);
    let mut cap: Vec<u16> = Vec::with_capacity(9);

    for _ in 0..9 {
        cap.push(0);
    }

    println!("Possible combinations between 1-9 with sum = 15: ");
    for j in 1..=9 {
        arr.push(j);
        rec(j, arr.clone(), &mut cap);
        arr.pop();
    }

    println!();
    println!("Occurence frequency of each number: ");
    println!("[Number: Frequency]");

    for (i, v) in cap.iter().enumerate() {
        print!("[{}: {v}], ", i + 1);
    }
}
